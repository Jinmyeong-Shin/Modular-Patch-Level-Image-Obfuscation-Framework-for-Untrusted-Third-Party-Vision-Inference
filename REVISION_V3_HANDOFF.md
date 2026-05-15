# Revision V3 Handoff

This note is for the next Codex/session after moving from the devcontainer to the local host. It captures the current revision state, the comparison argument around the Kiya/Kinoshita/Horio line of work, and the files that should be read first.

## Current Goal

Revise `paper/v3` after the experiment/artifact freeze. The paper should answer the reviewers by:

- toning down novelty claims from "groundbreaking" to an incremental but useful modular ViT-compatible engineering contribution;
- adding stronger adversarial evidence and a practical threat-model comparison;
- expanding task coverage beyond classification/OCR/detection where the artifacts support it;
- clarifying trusted deployment assumptions for Merkle attestation and obfuscator pools;
- tightening repetitive writing.

Do not make the paper sound personally targeted at any authors. The comparison should be written as a method-family and threat-model comparison.

## Read First

Start with these files:

- `outputs/revision_v3/README.md`
- `outputs/revision_v3/all_results.json`
- `outputs/revision_v3/adversarial-vit-cifar10_results.json`
- `outputs/revision_v3/horio-adversary-comparison-vit-cifar10_results.json`
- `outputs/revision_v3/permutation-family-key-recovery-comparison_results.json`
- `outputs/revision_v3/figures/direct_adversarial_comparison_sample.png`
- `outputs/revision_v3/figures/permutation-family-key-recovery-comparison_grid.png`
- `scripts/run_adversarial_scenarios.py`
- `scripts/run_horio_adversary_comparison.py`
- `scripts/run_permutation_family_comparison.py`
- `src/vit_obfuscation/attacks/horio_baseline.py`
- `paper/v3/access.tex`
- `paper/v3/references.bib`

The canonical experimental outputs are only under `outputs/revision_v3/`. Older outputs outside that directory were intentionally removed or should not be cited.

## Fun Status, Professionally Framed

The current artifact set includes a broad comparison against a Kiya-centered reversible image-encryption research line. In the comparison artifact, there are 17 relevant papers/method targets:

- Hitoshi Kiya appears in all 17.
- Yuma Kinoshita appears in 4.
- Kouki Horio appears in 1, the 2024 restricted random permutation ViT paper.

The important technical result is not "embarrass the authors." It is:

> Prior reversible perceptual-encryption approaches can make ciphertext images look visually degraded, but their security depends heavily on key secrecy and attack model. Under known-key or calibrated chosen-plaintext settings, reconstruction is exact. The proposed method is not a lossless permutation and does not expose a closed-form inverse; even a white-box optimization attack recovers only a degraded approximation.

This is the paper-safe way to use the result.

## Kiya/Kinoshita/Horio-Adjacent Paper Targets

The comparison artifact maps 17 papers into 7 representative reversible transform families:

1. `sirichotedumrong2019pixel` -> pixel negative/channel shuffle
2. `kawamura2020etc` -> EtC block scrambling
3. `maungmaung2022isotropic` -> EtC block scrambling
4. `qi2022convmixer_adaptive_permutation` -> EtC/adaptive permutation
5. `iijima2022convmixer_model_encryption` -> patch-local pixel shuffle
6. `qi2022vit` -> ViT block+pixel permutation
7. `kiya2023image_model` -> patch-local pixel shuffle
8. `kiya2022segmentation` -> patch-local pixel shuffle
9. `hamano2023jpeg_etc_vit` -> EtC/JPEG ViT
10. `aso2023random_orthogonal_convmixer` -> block-wise random orthogonal transform
11. `nagamori2023federated_vit` -> ViT block+pixel permutation
12. `nagamori2024domain_adaptation` -> ViT block+pixel permutation
13. `kiya2023reliable_vit` -> ViT block+pixel permutation
14. `aso2024disposable` -> restricted per-image/key permutation
15. `horio2024restricted` -> restricted random block/pixel permutation
16. `hirose2025no_key_management` -> restricted per-image/key permutation
17. `lin2024convmixer` -> EtC block scrambling

Before final manuscript submission, verify citations/metadata in `references.bib` and add missing BibTeX entries for any of the above that are cited in text.

## Key Comparison Numbers

From `outputs/revision_v3/permutation-family-key-recovery-comparison_results.json`:

| Family | Direct SSIM | Direct PSNR | Direct MSE | Known/Chosen-Plaintext Recovery |
|---|---:|---:|---:|---|
| Proposed obfuscator | 0.0022 | 10.35 | 0.3694 | no closed-form exact inverse |
| PE pixel negative/channel shuffle | 0.0076 | 8.41 | 0.5774 | SSIM 1.0, PSNR inf, MSE 0 |
| EtC block scrambling | 0.0281 | 8.43 | 0.5742 | SSIM 1.0, PSNR inf, MSE 0 |
| Block orthogonal secret key | 0.0009 | 8.32 | 0.5891 | SSIM 1.0, PSNR inf, MSE 0 |
| ViT block+pixel permutation | 0.0279 | 8.55 | 0.5590 | SSIM 1.0, PSNR inf, MSE 0 |
| Patch-local pixel shuffle | 0.1462 | 13.63 | 0.1734 | SSIM 1.0, PSNR inf, MSE 0 |
| Horio restricted `Nbs=120,Nps=500` | 0.2653 | 12.22 | 0.2400 | SSIM 1.0, PSNR inf, MSE 0 |
| Restricted per-image/key permutation | 0.3956 | 14.00 | 0.1594 | SSIM 1.0, PSNR inf, MSE 0 |

From `outputs/revision_v3/adversarial-vit-cifar10_results.json`:

- Proposed direct obfuscated: SSIM 0.0022, PSNR 10.35, MSE 0.3694.
- MI-FGSM: SSIM 0.1192, PSNR 11.50, MSE 0.2830.
- L-BFGS exact-secret white-box: SSIM 0.4598, PSNR 17.93, MSE 0.0645.
- Adversarial VAE: SSIM 0.0656, PSNR 11.61, MSE 0.2764.
- CycleGAN: SSIM 0.0438, PSNR 10.40, MSE 0.3645.
- Side channel: frequency correlation 0.0019, spatial autocorrelation -0.1885, histogram KL 2.0184, mutual information 0.0407.

Interpretation for paper:

- L-BFGS is a deliberately worst-case white-box attack: clean distribution, exact secret, and direct optimization access. Even there, reconstruction is not exact and remains visibly degraded.
- Permutation-family baselines are exact under known-key/chosen-plaintext calibration because they are reversible transforms.
- Per-image/disposable-key variants reduce cross-image key reuse but do not change the fact that a compromised/calibrated target key gives exact recovery for that target.

## Task Coverage Status

Use claim gating. Only make strong claims where artifacts are strong.

Strong / usable:

- Classification: existing CIFAR/medical classification artifacts under `outputs/revision_v3/`.
- CLIP zero-shot CIFAR-10: clean accuracy 0.8738, obfuscated 0.8500.
- Image-text retrieval, Flickr30k with CLIP: image-to-text R@1 clean 0.84, obfuscated 0.81; text-to-image R@1 clean 0.795, obfuscated 0.77.
- Object detection, after positional embedding/interpolation fixes:
  - YOLOS COCO: clean mAP 0.2709, obfuscated mAP 0.2329.
  - OWLViT COCO: clean mAP 0.2470, obfuscated mAP 0.2144.
- Industrial anomaly detection, MVTec AD with CLIP:
  - clean image AUROC 1.0, pixel AUROC 0.9694, PRO 0.8267.
  - obfuscated image AUROC 0.7168, pixel AUROC 0.9414, PRO 0.6931.

Weak / limitations only:

- Medical segmentation: Dice 0.0, IoU 0.0 for clean and obfuscated. Pixel accuracy is high because masks are mostly background. Do not claim segmentation success from this artifact.
- Captioning: BLIP Flickr30k BLEU is low and obfuscated is slightly higher than clean due to small/sample noise. Mention only as exploratory or omit from main claims.

## Paper Revision Instructions

Recommended order for `paper/v3/access.tex`:

1. Abstract:
   - Formal IEEE style.
   - Tone down claims.
   - Include PSNR/SSIM explicitly, per Reviewer 3.
   - Mention "empirically resistant under tested attacks" rather than "guarantees non-invertibility."

2. Introduction/contributions:
   - Reframe novelty as modular ViT-compatible obfuscation/deobfuscation engineering plus empirical privacy evidence.
   - Avoid saying the method is fundamentally new cryptography.
   - State incremental relation to decoder-free/permutation-based methods.

3. Related work:
   - Add a compact subsection on reversible perceptual encryption for ViTs and Kiya/Kinoshita/Horio-family methods.
   - Add broader privacy-preserving ViT/multimodal and adversarial reconstruction literature.
   - Keep tone neutral: "reversible by construction under key exposure" is enough.

4. Threat model/security:
   - Define black-box, gray-box, and white-box cases.
   - Explicitly separate ciphertext-only visual privacy from known-key/chosen-plaintext recovery.
   - Clarify Merkle attestation assumptions: it verifies declared configuration membership, not faithful runtime execution in a fully malicious cloud unless paired with hardware/other trust anchors.

5. Experiments:
   - Use only `outputs/revision_v3/*.json`.
   - Add "Additional Task Coverage" with retrieval, OD, and industrial anomaly detection.
   - Keep medical segmentation and captioning in limitations/future work unless rerun with stronger artifacts.
   - Add adversarial comparison table using proposed vs Horio/reversible-family numbers.

6. Limitations:
   - Trusted setup and Merkle assumptions.
   - No formal proof of non-invertibility.
   - Segmentation/captioning still exploratory.
   - Strong privacy claim is empirical and attack-bounded.

## Suggested Paper-Safe Phrasing

Use:

> The proposed obfuscator is not a lossless permutation and does not provide a closed-form inverse. In contrast, several prior perceptual-encryption approaches are reversible by design: when the secret key is known, or when a reused permutation key is calibrated through chosen plaintexts, exact reconstruction is possible. Therefore, their privacy guarantees and ours should be compared under explicitly separated threat models.

Avoid:

> This destroys their research.

Also avoid naming suspected reviewers or implying reviewer identity in the manuscript.

## Repo Transfer Notes

The code/artifact commit `5ce633c` has already been pushed to `origin/main` with the revision-v3 experiment code and JSON/figure artifacts. Large `.pt` checkpoints are intentionally not tracked. The local devcontainer currently has extra LaTeX build byproducts; do not rely on those for the handoff.

After moving to local host:

1. Pull latest `main`.
2. Read this handoff and the artifact JSONs listed above.
3. Continue editing `paper/v3/access.tex`.
4. Before final submission, ensure every table value in the paper is traceable to `outputs/revision_v3/`.
