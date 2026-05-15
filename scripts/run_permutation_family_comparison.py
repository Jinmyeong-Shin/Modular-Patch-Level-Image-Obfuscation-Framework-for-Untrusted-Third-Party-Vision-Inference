from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from run_adversarial_scenarios import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    _load_obfuscator,
    _load_unit_images,
    _processor_image_size,
    _save_grid,
)
from vit_obfuscation.attacks.evaluate_attacks import compute_psnr, compute_ssim  # noqa: E402
from vit_obfuscation.attacks.horio_baseline import (  # noqa: E402
    HorioPermutationConfig,
    HorioRestrictedPermutationObfuscator,
)
from vit_obfuscation.attacks.side_channel import side_channel_analysis  # noqa: E402
from vit_obfuscation.config.experiment import ExperimentConfig  # noqa: E402
from vit_obfuscation.outputs.manifest import build_manifest  # noqa: E402


DEFAULT_CONFIG = "configs/experiments/vit_cifar10.yaml"


@dataclass(frozen=True)
class PaperTarget:
    key: str
    title: str
    authors: str
    year: int
    source_url: str
    scheme_id: str
    notes: str


class PixelNegativeChannelShuffle(nn.Module):
    """Representative pixel-based PE family: sign flip plus channel shuffle."""

    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        seed: int = 42,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        height, width = image_size
        num_pixels = height * width
        generator = torch.Generator().manual_seed(seed)

        perms = []
        inv_perms = []
        for _ in range(num_pixels):
            perm = torch.randperm(num_channels, generator=generator)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(num_channels)
            perms.append(perm)
            inv_perms.append(inv)

        sign = torch.randint(
            0,
            2,
            (num_pixels, num_channels),
            generator=generator,
            dtype=torch.float32,
        )
        sign = sign * 2.0 - 1.0

        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.register_buffer("channel_permutation", torch.stack(perms))
        self.register_buffer("inverse_channel_permutation", torch.stack(inv_perms))
        self.register_buffer("sign_mask", sign)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        if (channels, height, width) != (self.num_channels, self.height, self.width):
            raise ValueError("input shape does not match transform configuration")

        flat = images.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        permuted = torch.gather(
            flat,
            2,
            self.channel_permutation.unsqueeze(0).expand(batch, -1, -1),
        )
        encrypted = permuted * self.sign_mask.unsqueeze(0)
        return encrypted.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    def inverse(self, encrypted_images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = encrypted_images.shape
        flat = encrypted_images.permute(0, 2, 3, 1).reshape(
            batch,
            height * width,
            channels,
        )
        unsigned = flat * self.sign_mask.unsqueeze(0)
        recovered = torch.gather(
            unsigned,
            2,
            self.inverse_channel_permutation.unsqueeze(0).expand(batch, -1, -1),
        )
        return recovered.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class EtCBlockScramble(nn.Module):
    """Representative EtC/block-wise family with reversible block transforms."""

    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        patch_size: int = 16,
        seed: int = 42,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        height, width = image_size
        if height % patch_size or width % patch_size:
            raise ValueError("image_size must be divisible by patch_size")

        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.blocks_h = height // patch_size
        self.blocks_w = width // patch_size
        num_blocks = self.blocks_h * self.blocks_w
        generator = torch.Generator().manual_seed(seed)

        block_perm = torch.randperm(num_blocks, generator=generator)
        inverse_block_perm = torch.empty_like(block_perm)
        inverse_block_perm[block_perm] = torch.arange(num_blocks)

        channel_perms = []
        inverse_channel_perms = []
        for _ in range(num_blocks):
            perm = torch.randperm(num_channels, generator=generator)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(num_channels)
            channel_perms.append(perm)
            inverse_channel_perms.append(inv)

        self.register_buffer("block_permutation", block_perm)
        self.register_buffer("inverse_block_permutation", inverse_block_perm)
        self.register_buffer("channel_permutation", torch.stack(channel_perms))
        self.register_buffer(
            "inverse_channel_permutation",
            torch.stack(inverse_channel_perms),
        )
        self.register_buffer(
            "rotations",
            torch.randint(0, 4, (num_blocks,), generator=generator),
        )
        self.register_buffer(
            "flip_h",
            torch.randint(0, 2, (num_blocks,), generator=generator, dtype=torch.bool),
        )
        self.register_buffer(
            "flip_w",
            torch.randint(0, 2, (num_blocks,), generator=generator, dtype=torch.bool),
        )
        sign = torch.randint(
            0,
            2,
            (num_blocks, num_channels),
            generator=generator,
            dtype=torch.float32,
        )
        self.register_buffer("sign_mask", sign * 2.0 - 1.0)

    @property
    def num_blocks(self) -> int:
        return int(self.block_permutation.numel())

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        return (
            images.reshape(
                batch,
                channels,
                self.blocks_h,
                self.patch_size,
                self.blocks_w,
                self.patch_size,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, self.num_blocks, channels, self.patch_size, self.patch_size)
        )

    def _unpatchify(self, blocks: torch.Tensor) -> torch.Tensor:
        batch = blocks.shape[0]
        return (
            blocks.reshape(
                batch,
                self.blocks_h,
                self.blocks_w,
                self.num_channels,
                self.patch_size,
                self.patch_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, self.num_channels, self.height, self.width)
        )

    def _encrypt_block(self, block: torch.Tensor, block_index: int) -> torch.Tensor:
        out = block[:, self.channel_permutation[block_index]]
        out = out * self.sign_mask[block_index].view(1, -1, 1, 1)
        if bool(self.flip_h[block_index]):
            out = torch.flip(out, dims=(-2,))
        if bool(self.flip_w[block_index]):
            out = torch.flip(out, dims=(-1,))
        rotation = int(self.rotations[block_index].item())
        if rotation:
            out = torch.rot90(out, rotation, dims=(-2, -1))
        return out

    def _decrypt_block(self, block: torch.Tensor, block_index: int) -> torch.Tensor:
        out = block
        rotation = int(self.rotations[block_index].item())
        if rotation:
            out = torch.rot90(out, (-rotation) % 4, dims=(-2, -1))
        if bool(self.flip_w[block_index]):
            out = torch.flip(out, dims=(-1,))
        if bool(self.flip_h[block_index]):
            out = torch.flip(out, dims=(-2,))
        out = out * self.sign_mask[block_index].view(1, -1, 1, 1)
        return out[:, self.inverse_channel_permutation[block_index]]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        blocks = self._patchify(images)
        encrypted = torch.empty_like(blocks)
        for encrypted_index in range(self.num_blocks):
            source_index = int(self.block_permutation[encrypted_index].item())
            encrypted[:, encrypted_index] = self._encrypt_block(
                blocks[:, source_index],
                encrypted_index,
            )
        return self._unpatchify(encrypted)

    def inverse(self, encrypted_images: torch.Tensor) -> torch.Tensor:
        blocks = self._patchify(encrypted_images)
        recovered = torch.empty_like(blocks)
        for encrypted_index in range(self.num_blocks):
            source_index = int(self.block_permutation[encrypted_index].item())
            recovered[:, source_index] = self._decrypt_block(
                blocks[:, encrypted_index],
                encrypted_index,
            )
        return self._unpatchify(recovered)


class BlockOrthogonalTransform(nn.Module):
    """Representative block-wise random orthogonal secret-key transform."""

    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        patch_size: int = 16,
        seed: int = 42,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        height, width = image_size
        if height % patch_size or width % patch_size:
            raise ValueError("image_size must be divisible by patch_size")

        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.blocks_h = height // patch_size
        self.blocks_w = width // patch_size
        vector_dim = patch_size * patch_size * num_channels
        generator = torch.Generator().manual_seed(seed)
        random_matrix = torch.randn(vector_dim, vector_dim, generator=generator)
        q, _ = torch.linalg.qr(random_matrix)
        self.register_buffer("orthogonal_matrix", q)

    @property
    def num_blocks(self) -> int:
        return self.blocks_h * self.blocks_w

    @property
    def vector_dim(self) -> int:
        return self.patch_size * self.patch_size * self.num_channels

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        return (
            images.reshape(
                batch,
                channels,
                self.blocks_h,
                self.patch_size,
                self.blocks_w,
                self.patch_size,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, self.num_blocks, self.vector_dim)
        )

    def _unpatchify(self, blocks: torch.Tensor) -> torch.Tensor:
        batch = blocks.shape[0]
        return (
            blocks.reshape(
                batch,
                self.blocks_h,
                self.blocks_w,
                self.num_channels,
                self.patch_size,
                self.patch_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, self.num_channels, self.height, self.width)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        blocks = self._patchify(images)
        transformed = torch.matmul(blocks, self.orthogonal_matrix.T)
        return self._unpatchify(transformed)

    def inverse(self, encrypted_images: torch.Tensor) -> torch.Tensor:
        blocks = self._patchify(encrypted_images)
        recovered = torch.matmul(blocks, self.orthogonal_matrix)
        return self._unpatchify(recovered)


def _json_number(value: float) -> float | str:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return float(value)


def _metric_row(images: torch.Tensor, reference: torch.Tensor) -> dict:
    mse = F.mse_loss(images, reference).item()
    if abs(mse) < 1e-10:
        return {
            "ssim": 1.0,
            "psnr": "inf",
            "mse": 0.0,
        }

    ssim = compute_ssim(images, reference, data_range=2.0)
    if ssim > 1.0 and ssim - 1.0 < 1e-5:
        ssim = 1.0

    return {
        "ssim": _json_number(ssim),
        "psnr": _json_number(compute_psnr(images, reference, data_range=2.0)),
        "mse": _json_number(mse),
    }


def _side_channel_dict(original: torch.Tensor, obfuscated: torch.Tensor) -> dict:
    side = side_channel_analysis(original, obfuscated)
    return {
        "frequency_correlation": side.frequency_correlation,
        "spatial_autocorrelation": side.spatial_autocorrelation,
        "histogram_kl_divergence": side.histogram_kl_divergence,
        "mutual_information": side.mutual_information,
    }


def _build_paper_targets() -> list[PaperTarget]:
    return [
        PaperTarget(
            key="sirichotedumrong2019pixel",
            title="Pixel-Based Image Encryption without Key Management for Privacy-Preserving Deep Neural Networks",
            authors="Sirichotedumrong, Kinoshita, and Kiya",
            year=2019,
            source_url="https://doi.org/10.1109/ACCESS.2019.2959017",
            scheme_id="pixel_negative_channel_shuffle",
            notes="Pixel-wise negative-positive transformation and color-component shuffling family.",
        ),
        PaperTarget(
            key="kawamura2020etc",
            title="A Privacy-Preserving Machine Learning Scheme Using EtC Images",
            authors="Kawamura, Kinoshita, Nakachi, Shiota, and Kiya",
            year=2020,
            source_url="https://doi.org/10.1587/transfun.2020SMP0022",
            scheme_id="etc_block_scramble",
            notes="Encryption-then-compression block scrambling family.",
        ),
        PaperTarget(
            key="maungmaung2022isotropic",
            title="Privacy-Preserving Image Classification Using an Isotropic Network",
            authors="MaungMaung and Kiya",
            year=2022,
            source_url="https://arxiv.org/abs/2204.07707",
            scheme_id="etc_block_scramble",
            notes="EtC/block-wise encryption applied to isotropic networks such as ViT and ConvMixer.",
        ),
        PaperTarget(
            key="qi2022convmixer_adaptive_permutation",
            title="Privacy-Preserving Image Classification Using ConvMixer with Adaptive Permutation Matrix",
            authors="Qi, MaungMaung, and Kiya",
            year=2022,
            source_url="https://arxiv.org/abs/2208.02556",
            scheme_id="etc_block_scramble",
            notes="ConvMixer adaptation of block-wise scrambled image encryption with an adaptive permutation matrix.",
        ),
        PaperTarget(
            key="iijima2022convmixer_model_encryption",
            title="An Encryption Method of ConvMixer Models without Performance Degradation",
            authors="Iijima and Kiya",
            year=2022,
            source_url="https://arxiv.org/abs/2207.11939",
            scheme_id="patch_pixel_shuffle_only",
            notes="Secret-key image/model transformation for ConvMixer; image-side transform is reversible under the key.",
        ),
        PaperTarget(
            key="qi2022vit",
            title="Privacy-Preserving Image Classification Using Vision Transformer",
            authors="Qi, MaungMaung, Kinoshita, and Kiya",
            year=2022,
            source_url="https://arxiv.org/abs/2205.12041",
            scheme_id="vit_block_pixel_permutation",
            notes="Block scrambling plus in-block pixel-position shuffling for ViT-B/16.",
        ),
        PaperTarget(
            key="kiya2023image_model",
            title="Image and Model Transformation with Secret Key for Vision Transformer",
            authors="Kiya, Iijima, MaungMaung, and Kinoshita",
            year=2023,
            source_url="https://doi.org/10.1587/transinf.2022MUI0001",
            scheme_id="patch_pixel_shuffle_only",
            notes="Secret-key model/image transformation; image-side operation is patch-level pixel shuffling.",
        ),
        PaperTarget(
            key="kiya2022segmentation",
            title="Privacy-Preserving Semantic Segmentation Using Vision Transformer",
            authors="Kiya, Nagamori, Imaizumi, and Shiota",
            year=2022,
            source_url="https://www.mdpi.com/2313-433X/8/9/233",
            scheme_id="patch_pixel_shuffle_only",
            notes="SETR-compatible patch pixel shuffling; preserves patch positions for dense prediction.",
        ),
        PaperTarget(
            key="hamano2023jpeg_etc_vit",
            title="Effects of JPEG Compression on Vision Transformer Image Classification for Encryption-then-Compression Images",
            authors="Hamano, Imaizumi, and Kiya",
            year=2023,
            source_url="https://www.mdpi.com/1424-8220/23/7/3400",
            scheme_id="etc_block_scramble",
            notes="Studies JPEG-compressed EtC images for ViT classification; image transform remains reversible before compression.",
        ),
        PaperTarget(
            key="aso2023random_orthogonal_convmixer",
            title="A Privacy Preserving Method with a Random Orthogonal Matrix for ConvMixer Models",
            authors="Aso, Chuman, and Kiya",
            year=2023,
            source_url="https://arxiv.org/abs/2301.03843",
            scheme_id="block_orthogonal_secret_key",
            notes="Block-wise random orthogonal matrix transform; included as a reversible linear-key baseline, not a pure permutation.",
        ),
        PaperTarget(
            key="nagamori2023federated_vit",
            title="Combined Use of Federated Learning and Image Encryption for Privacy-Preserving Image Classification with Vision Transformer",
            authors="Nagamori and Kiya",
            year=2023,
            source_url="https://arxiv.org/abs/2301.09255",
            scheme_id="vit_block_pixel_permutation",
            notes="Federated-learning setting using the same reversible ViT-oriented image encryption family.",
        ),
        PaperTarget(
            key="nagamori2024domain_adaptation",
            title="Efficient Fine-Tuning with Domain Adaptation for Privacy-Preserving Vision Transformer",
            authors="Nagamori, Shiota, and Kiya",
            year=2024,
            source_url="https://arxiv.org/abs/2401.05126",
            scheme_id="vit_block_pixel_permutation",
            notes="Same block scrambling plus pixel shuffling, with domain adaptation for utility.",
        ),
        PaperTarget(
            key="kiya2023reliable_vit",
            title="Block-Wise Encryption for Reliable Vision Transformer Models",
            authors="Kiya, Iijima, and Nagamori",
            year=2023,
            source_url="https://arxiv.org/abs/2308.07612",
            scheme_id="vit_block_pixel_permutation",
            notes="Survey/application paper around block-wise encryption for ViT models.",
        ),
        PaperTarget(
            key="aso2024disposable",
            title="Disposable-key-based Image Encryption for Collaborative Learning of Vision Transformer",
            authors="Aso, Shiota, and Kiya",
            year=2024,
            source_url="https://arxiv.org/abs/2408.05737",
            scheme_id="restricted_per_image_permutation",
            notes="Restricted random block/pixel permutation matrices with independent keys per image/client.",
        ),
        PaperTarget(
            key="horio2024restricted",
            title="Privacy-Preserving Vision Transformer Using Images Encrypted with Restricted Random Permutation Matrices",
            authors="Horio, Nishikawa, and Kiya",
            year=2024,
            source_url="https://arxiv.org/abs/2408.08529",
            scheme_id="restricted_horio_high_accuracy",
            notes="Restricted block and pixel random permutation matrices.",
        ),
        PaperTarget(
            key="hirose2025no_key_management",
            title="Learnable Image Encryption Without Key Management for Privacy-Preserving Vision Transformer",
            authors="Hirose, Imaizumi, and Kiya",
            year=2025,
            source_url="https://doi.org/10.1109/ACCESS.2025.3635235",
            scheme_id="restricted_per_image_permutation",
            notes="Independent per-client/per-image key family; calibration must match the target key.",
        ),
        PaperTarget(
            key="lin2024convmixer",
            title="Privacy-Preserving ConvMixer Without Any Accuracy Degradation Using Compressible Encrypted Images",
            authors="Lin, Imaizumi, and Kiya",
            year=2024,
            source_url="https://www.mdpi.com/2078-2489/15/11/723",
            scheme_id="etc_block_scramble",
            notes="Compressible encrypted image family adapted for ConvMixer.",
        ),
    ]


def _reported_attack_numbers() -> dict:
    return {
        "qi2022vit_ssim_values_from_figure": {
            "source_url": "https://arxiv.org/abs/2205.12041",
            "metric": "SSIM; higher means more visual recovery",
            "rows": {
                "LE": {
                    "encrypted": 0.006,
                    "FR_attack": 0.017,
                    "GAN_attack": 0.774,
                    "ITN_attack": 0.407,
                },
                "EtC": {
                    "encrypted": 0.001,
                    "FR_attack": 0.010,
                    "GAN_attack": 0.031,
                    "ITN_attack": 0.529,
                },
                "ELE": {
                    "encrypted": 0.061,
                    "FR_attack": 0.001,
                    "GAN_attack": 0.021,
                    "ITN_attack": 0.001,
                },
                "PE": {
                    "encrypted": 0.001,
                    "FR_attack": 0.001,
                    "GAN_attack": 0.010,
                    "ITN_attack": 0.086,
                },
                "ViT_block_pixel_permutation": {
                    "encrypted": 0.123,
                    "FR_attack": 0.035,
                    "GAN_attack": 0.043,
                    "ITN_attack": 0.117,
                },
            },
        },
        "chuman2023_jigsaw_attack": {
            "source_url": "https://www.mdpi.com/2078-2489/14/6/311",
            "note": (
                "Reports that a jigsaw-puzzle-based ciphertext-only attack can "
                "restore almost all visual information from ViT-oriented "
                "block-scrambling/pixel-shuffling encrypted images."
            ),
        },
    }


def _scheme_modules(
    image_size: tuple[int, int],
    seed: int,
) -> dict[str, tuple[nn.Module, str, str]]:
    return {
        "pixel_negative_channel_shuffle": (
            PixelNegativeChannelShuffle(image_size=image_size, seed=seed),
            "PE-style pixel negative/channel shuffle",
            "exact with known key; chosen plaintext can recover per-pixel signs/channel order",
        ),
        "etc_block_scramble": (
            EtCBlockScramble(image_size=image_size, seed=seed),
            "EtC-style block scrambling",
            "exact with known key; chosen plaintext can recover block/pixel/channel operations",
        ),
        "block_orthogonal_secret_key": (
            BlockOrthogonalTransform(image_size=image_size, seed=seed),
            "block-wise random orthogonal transform",
            "exact with known key; chosen plaintext basis probes can recover the orthogonal matrix",
        ),
        "vit_block_pixel_permutation": (
            HorioRestrictedPermutationObfuscator(
                image_size=image_size,
                config=HorioPermutationConfig(
                    patch_size=16,
                    fixed_blocks=0,
                    fixed_pixels=0,
                    seed=seed,
                ),
            ),
            "unrestricted ViT block+pixel permutation",
            "one unique calibration image recovers the global permutation when the same key is reused",
        ),
        "patch_pixel_shuffle_only": (
            HorioRestrictedPermutationObfuscator(
                image_size=image_size,
                config=HorioPermutationConfig(
                    patch_size=16,
                    fixed_blocks=196,
                    fixed_pixels=0,
                    seed=seed,
                ),
            ),
            "patch-local pixel permutation only",
            "one unique calibration image recovers the in-patch permutation when the same key is reused",
        ),
        "restricted_horio_high_accuracy": (
            HorioRestrictedPermutationObfuscator(
                image_size=image_size,
                config=HorioPermutationConfig(
                    patch_size=16,
                    fixed_blocks=120,
                    fixed_pixels=500,
                    seed=seed,
                ),
            ),
            "Horio restricted setting Nbs=120,Nps=500",
            "one unique calibration image recovers the restricted permutation when the same key is reused",
        ),
        "restricted_per_image_permutation": (
            HorioRestrictedPermutationObfuscator(
                image_size=image_size,
                config=HorioPermutationConfig(
                    patch_size=16,
                    fixed_blocks=147,
                    fixed_pixels=576,
                    seed=seed,
                ),
            ),
            "restricted per-image/key permutation example Nbs=147,Nps=576",
            "exact per target key; per-image disposable keys prevent reuse but not per-image key compromise",
        ),
    }


def run_comparison(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    config = ExperimentConfig.from_yaml(args.config)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    image_size = _processor_image_size(config.model.hf_model_name_or_path)
    original = _load_unit_images(
        config.embedding_training.training_dataset,
        image_size,
        args.num_eval_images,
    ).to(device)

    schemes = _scheme_modules(image_size, args.seed)
    paper_targets = _build_paper_targets()

    scheme_results = []
    figure_rows: list[tuple[str, torch.Tensor]] = [("original", original)]

    try:
        proposed_obfuscator, checkpoint_path, _ = _load_obfuscator(config, output_dir)
        proposed_obfuscator = proposed_obfuscator.to(device).eval()
        with torch.no_grad():
            proposed_obfuscated = proposed_obfuscator(original)
        proposed_direct = _metric_row(proposed_obfuscated, original)
        figure_rows.append(("proposed obfuscated", proposed_obfuscated))
    except FileNotFoundError:
        checkpoint_path = None
        proposed_direct = None

    for scheme_id, (module, label, key_recovery_note) in schemes.items():
        module = module.to(device).eval()
        with torch.no_grad():
            obfuscated = module(original)
            recovered = module.inverse(obfuscated)
        direct = _metric_row(obfuscated, original)
        recovered_metrics = _metric_row(recovered, original)
        side = _side_channel_dict(original, obfuscated)
        mapped_papers = [
            paper.key for paper in paper_targets if paper.scheme_id == scheme_id
        ]

        scheme_results.append(
            {
                "scheme_id": scheme_id,
                "label": label,
                "mapped_papers": mapped_papers,
                "direct_obfuscated_quality": direct,
                "exact_key_or_chosen_plaintext_recovery": recovered_metrics,
                "key_recovery_note": key_recovery_note,
                "side_channel": side,
            }
        )
        figure_rows.append((label, obfuscated))

    result_name = "permutation-family-key-recovery-comparison"
    result_path = output_dir / f"{result_name}_results.json"
    grid_path = output_dir / "figures" / f"{result_name}_grid.png"
    _save_grid(figure_rows, grid_path, max_images=min(4, args.num_eval_images))

    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": result_name,
        "task": "key_recovery_and_direct_obfuscation_quality_comparison",
        "base_experiment": config.name,
        "image_size": list(image_size),
        "sample_counts": {"eval": args.num_eval_images},
        "proposed_reference": {
            "direct_obfuscated_quality": proposed_direct,
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "white_box_l_bfgs_note": (
                "See adversarial-vit-cifar10_results.json for exact-secret "
                "optimization results; the proposed transform is not a lossless "
                "permutation and does not have a closed-form inverse."
            ),
        },
        "scheme_results": scheme_results,
        "paper_targets": [
            {
                "key": paper.key,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "source_url": paper.source_url,
                "scheme_id": paper.scheme_id,
                "notes": paper.notes,
            }
            for paper in paper_targets
        ],
        "reported_other_attack_numbers": _reported_attack_numbers(),
        "interpretation": [
            (
                "All listed permutation/negative-positive/channel-shuffle families "
                "are deterministic reversible transforms. With the correct key, "
                "reconstruction is exact; with reused pure permutation keys, a "
                "chosen calibration image can recover the inverse permutation."
            ),
            (
                "Per-image or disposable-key variants reduce cross-image key reuse, "
                "but the protected image for a compromised or calibrated key remains "
                "exactly recoverable."
            ),
            (
                "Direct obfuscated-image quality should be reported separately from "
                "known-key/chosen-plaintext recovery because ciphertext-only visual "
                "quality can look low while exact-key failure remains total."
            ),
        ],
        "figure": str(grid_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with result_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)

    manifest_config = copy.deepcopy(config)
    manifest_config.name = result_name
    manifest = build_manifest(
        config=manifest_config,
        result_file=result_path,
        status="success",
    )
    manifest_path = output_dir / f"{result_name}_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Kiya/Kinoshita-family permutation encryption schemes under "
            "direct leakage and exact key/chosen-plaintext recovery."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-eval-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = run_comparison(args)
    print(f"[done] wrote {result['experiment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
