from __future__ import annotations

from collections import Counter
from math import exp, log

import torch
from tqdm import tqdm

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import ExperimentConfig
from ..datasets.loader import load_classification_dataset
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.types import CaptioningEvalOutput
from .base import BaseTask
from .feature_utils import collate_images, first_text, maybe_limit_dataset, normalize_text


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def _sentence_bleu(candidate: str, reference: str, max_n: int) -> float:
    cand = normalize_text(candidate)
    ref = normalize_text(reference)
    if not cand or not ref:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_counts = _ngram_counts(cand, n)
        ref_counts = _ngram_counts(ref, n)
        if not cand_counts:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_counts[ngram]) for ngram, count in cand_counts.items())
        precisions.append((overlap + 1) / (sum(cand_counts.values()) + 1))
    brevity = 1.0 if len(cand) > len(ref) else exp(1 - len(ref) / max(len(cand), 1))
    return float(brevity * exp(sum(log(p) for p in precisions) / max_n))


class ImageCaptioningTask(BaseTask):
    """Image captioning with an open VLM and swappable visual embeddings."""

    def forward(self, images, with_obfuscation: bool = False, **kwargs):
        model = self.adapter.model
        inputs = self.processor(images=images, return_tensors="pt")
        device = next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)
        if with_obfuscation:
            pixel_values = self.obfuscator(pixel_values)
            with self.adapter.obfuscation_mode(self.obf_embedding):
                return model.generate(pixel_values=pixel_values, **kwargs)
        return model.generate(pixel_values=pixel_values, **kwargs)

    def train_task(self) -> None:
        pass

    def evaluate(self, with_obfuscation: bool = False) -> CaptioningEvalOutput:
        cfg = self.config
        text_column = cfg.dataset.text_column or cfg.dataset.label_column
        _, eval_dataset = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
            subset=cfg.dataset.subset,
        )
        eval_dataset = maybe_limit_dataset(eval_dataset, cfg.evaluation.max_samples)

        def collate_fn(batch):
            return {
                cfg.dataset.input_column: collate_images(batch, cfg.dataset.input_column),
                text_column: [first_text(item[text_column]) for item in batch],
            }

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.adapter.model.to(device).eval()
        if with_obfuscation:
            self.obfuscation_modules_to(device)

        bleu1 = []
        bleu4 = []
        exact = []
        for batch in tqdm(dataloader, desc="Evaluating image captioning"):
            with torch.no_grad():
                inputs = self.processor(
                    images=batch[cfg.dataset.input_column],
                    return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(device)
                if with_obfuscation:
                    pixel_values = self.obfuscator(pixel_values)
                    adapter = ModelAdapter(model, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)
                    generated = model.generate(
                        pixel_values=pixel_values,
                        max_new_tokens=cfg.evaluation.max_new_tokens,
                        num_beams=cfg.evaluation.num_beams,
                    )
                    adapter.swap_to_original()
                else:
                    generated = model.generate(
                        pixel_values=pixel_values,
                        max_new_tokens=cfg.evaluation.max_new_tokens,
                        num_beams=cfg.evaluation.num_beams,
                    )
                predictions = self.processor.batch_decode(
                    generated, skip_special_tokens=True
                )
                for prediction, reference in zip(predictions, batch[text_column]):
                    pred_norm = " ".join(normalize_text(prediction))
                    ref_norm = " ".join(normalize_text(reference))
                    bleu1.append(_sentence_bleu(prediction, reference, 1))
                    bleu4.append(_sentence_bleu(prediction, reference, 4))
                    exact.append(float(pred_norm == ref_norm))

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")
        self.adapter.model = model.cpu()

        return CaptioningEvalOutput(
            bleu1=sum(bleu1) / len(bleu1) if bleu1 else 0.0,
            bleu4=sum(bleu4) / len(bleu4) if bleu4 else 0.0,
            exact_match=sum(exact) / len(exact) if exact else 0.0,
        )
