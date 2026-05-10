from __future__ import annotations

import itertools
from typing import Any

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import ExperimentConfig
from ..datasets.loader import load_classification_dataset
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.types import ClassificationEvalOutput, ClassificationModelOutput
from .base import BaseTask


def _image_collate_fn(batch, input_column, label_column):
    """Collate that keeps images as a list (PIL can't be stacked)."""
    images = [item[input_column] for item in batch]
    labels = torch.tensor([item[label_column] for item in batch])
    return {input_column: images, label_column: labels}


class ClassificationTask(BaseTask):
    """ViT-based image classification with a learned linear head."""

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(adapter, obfuscator, obf_embedding, processor, config)

        vision_config = adapter.get_vision_config()
        num_classes = config.model.num_classes
        assert num_classes is not None, "num_classes is required for ClassificationTask"

        self.classifier = nn.Linear(vision_config.hidden_size, num_classes)

    def forward(
        self, images, with_obfuscation: bool = False, **kwargs
    ) -> ClassificationModelOutput:
        model = self.adapter.model
        inputs = self.process_images(images)
        device = next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)

        if with_obfuscation:
            obfuscated = self.obfuscator(pixel_values)
            self.adapter.swap_to_obfuscation(self.obf_embedding)
            vit_outputs = model(pixel_values=obfuscated)
            self.adapter.swap_to_original()
            logits = self.classifier(vit_outputs.pooler_output)
            return ClassificationModelOutput(
                obfuscated_images=obfuscated,
                model_outputs=vit_outputs,
                logits=logits,
            )
        else:
            vit_outputs = model(pixel_values=pixel_values)
            logits = self.classifier(vit_outputs.pooler_output)
            return ClassificationModelOutput(
                obfuscated_images=None,
                model_outputs=vit_outputs,
                logits=logits,
            )

    def train_task(self) -> None:
        """Train the classification head (and optionally the full model)."""
        cfg = self.config
        task_cfg = cfg.task_training
        if task_cfg is None:
            return

        train_dataset, _ = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
        )

        accelerator = accelerate.Accelerator()

        # Combine model + classifier parameters
        params = list(self.adapter.model.parameters()) + list(
            self.classifier.parameters()
        )
        optimizer = torch.optim.AdamW(params, lr=task_cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=task_cfg.learning_rate,
            total_steps=task_cfg.iterations,
        )

        collate = lambda batch: _image_collate_fn(
            batch, cfg.dataset.input_column, cfg.dataset.label_column
        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=task_cfg.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

        model = self.adapter.model
        classifier = self.classifier
        model, classifier, optimizer, scheduler, dataloader = accelerator.prepare(
            model,
            classifier,
            optimizer,
            scheduler,
            dataloader,
        )

        model.train()
        classifier.train()

        pbar = tqdm(
            itertools.islice(itertools.cycle(dataloader), task_cfg.iterations),
            total=task_cfg.iterations,
            desc="Training classifier",
        )

        for batch in pbar:
            optimizer.zero_grad()

            images = batch[cfg.dataset.input_column]
            labels = batch[cfg.dataset.label_column].to(accelerator.device)

            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(accelerator.device)

            vit_outputs = model(pixel_values=pixel_values)
            logits = classifier(vit_outputs.pooler_output)

            loss = F.cross_entropy(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.adapter.model = accelerator.unwrap_model(model).cpu()
        self.classifier = accelerator.unwrap_model(classifier).cpu()

    def evaluate(self, with_obfuscation: bool = False) -> ClassificationEvalOutput:
        cfg = self.config

        _, eval_dataset = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
        )

        accelerator = accelerate.Accelerator()

        collate = lambda batch: _image_collate_fn(
            batch, cfg.dataset.input_column, cfg.dataset.label_column
        )
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        model = self.adapter.model
        classifier = self.classifier
        model, classifier, dataloader = accelerator.prepare(
            model, classifier, dataloader
        )

        if with_obfuscation:
            self.obfuscation_modules_to(accelerator.device)

        all_logits = []
        all_labels = []

        model.eval()
        classifier.eval()

        for batch in tqdm(dataloader, desc="Evaluating"):
            with torch.no_grad():
                images = batch[cfg.dataset.input_column]
                labels = batch[cfg.dataset.label_column]

                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(accelerator.device)

                if with_obfuscation:
                    pixel_values = self.obfuscator(pixel_values)
                    unwrapped = accelerator.unwrap_model(model)
                    adapter = ModelAdapter(unwrapped, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)
                    vit_outputs = model(pixel_values=pixel_values)
                    adapter.swap_to_original()
                else:
                    vit_outputs = model(pixel_values=pixel_values)

                logits = classifier(vit_outputs.pooler_output)
                all_logits.append(logits)
                all_labels.append(labels)

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")

        logits_arr = torch.cat(all_logits, dim=0).cpu().numpy()
        labels_arr = torch.cat(all_labels, dim=0).cpu().numpy()
        preds = np.argmax(logits_arr, axis=1)

        return ClassificationEvalOutput(
            accuracy=accuracy_score(labels_arr, preds),
            recall=recall_score(labels_arr, preds, average="macro"),
            precision=precision_score(labels_arr, preds, average="macro"),
            f1=f1_score(labels_arr, preds, average="macro"),
        )


class ZeroShotClassificationTask(BaseTask):
    """CLIP-style zero-shot image classification."""

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(adapter, obfuscator, obf_embedding, processor, config)

    def forward(
        self, images, text: list[str], with_obfuscation: bool = False, **kwargs
    ):
        model = self.adapter.model
        inputs = self.processor(
            images=images,
            text=text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if with_obfuscation:
            obfuscated = self.obfuscator(inputs["pixel_values"])
            self.adapter.swap_to_obfuscation(self.obf_embedding)
            inputs["pixel_values"] = obfuscated
            outputs = model(**inputs)
            self.adapter.swap_to_original()
        else:
            outputs = model(**inputs)

        return outputs

    def train_task(self) -> None:
        pass  # Zero-shot: no task training

    def evaluate(self, with_obfuscation: bool = False) -> ClassificationEvalOutput:
        cfg = self.config

        _, eval_dataset = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
        )

        # Get label names for zero-shot
        id2label = cfg.dataset.id2label or cfg.model.id2label
        if id2label is None:
            raise ValueError("id2label mapping required for zero-shot classification")
        label_names = [name for _, name in sorted(id2label.items())]

        accelerator = accelerate.Accelerator()

        collate = lambda batch: _image_collate_fn(
            batch, cfg.dataset.input_column, cfg.dataset.label_column
        )
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        model = self.adapter.model
        model, dataloader = accelerator.prepare(model, dataloader)

        if with_obfuscation:
            self.obfuscation_modules_to(accelerator.device)

        all_logits = []
        all_labels = []

        model.eval()

        for batch in tqdm(dataloader, desc="Evaluating zero-shot"):
            with torch.no_grad():
                images = batch[cfg.dataset.input_column]
                labels = batch[cfg.dataset.label_column]

                inputs = self.processor(
                    images=images,
                    text=label_names,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

                if with_obfuscation:
                    obfuscated = self.obfuscator(inputs["pixel_values"])
                    unwrapped = accelerator.unwrap_model(model)
                    adapter = ModelAdapter(unwrapped, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)
                    inputs["pixel_values"] = obfuscated
                    outputs = model(**inputs)
                    adapter.swap_to_original()
                else:
                    outputs = model(**inputs)

                all_logits.append(outputs.logits_per_image)
                all_labels.append(labels)

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")

        logits_arr = torch.cat(all_logits, dim=0).cpu().numpy()
        labels_arr = torch.cat(all_labels, dim=0).cpu().numpy()
        preds = np.argmax(logits_arr, axis=1)

        return ClassificationEvalOutput(
            accuracy=accuracy_score(labels_arr, preds),
            recall=recall_score(labels_arr, preds, average="macro"),
            precision=precision_score(labels_arr, preds, average="macro"),
            f1=f1_score(labels_arr, preds, average="macro"),
        )
