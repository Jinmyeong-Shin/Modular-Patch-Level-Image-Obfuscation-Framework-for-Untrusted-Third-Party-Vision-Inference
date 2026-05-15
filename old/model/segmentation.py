from typing import Optional, Any, List
from dataclasses import dataclass

import copy
import itertools
import json

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import accelerate
try:
    # For torchmetrics >= v0.11.0
    from torchmetrics import JaccardIndex
except ImportError:
    # For torchmetrics < v0.11.0
    from torchmetrics.classification import JaccardIndex

from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor, CLIPSegForImageSegmentation, CLIPSegProcessor

from data.type import ImageInput

# The following are placeholders for data types that should exist in your project.
# You would need to create/define these in the respective `data` module files.
SegmentationDatasetConfigWithTrainingAndEvaluationSplits = Any

from data.model_config import ImageModelConfigWithObfuscation

from data.obfusccation_output import ObfuscationOutput
from data.model_output import ModelOutputWithObfuscation, SegmentationModelOutputWithObfuscation
from data.evaluation_output import SegmentationEvaluationOutput

from .base import BaseImageModelWithObfuscation


class ImageSegmentationModel(BaseImageModelWithObfuscation):
    """
    Image segmentation model based on a fine-tunable Transformer architecture like SegFormer.
    """
    def __init__(
        self, 
        config: ImageModelConfigWithObfuscation,
        pretrained: bool = True
    ) -> None:
        nn.Module.__init__(self)

        self.config = config

        if pretrained:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(config.hf_model_name_or_path)
        else:
            self.model = AutoModelForSemanticSegmentation.from_config(config.hf_model_config)

        self.processor = AutoImageProcessor.from_pretrained(config.hf_model_name_or_path)

        self.vision_config = self.model.config

        # Assuming a SegFormer-like model architecture
        self.original_embeddings = self.model.segformer.encoder.patch_embeddings

        self.__post_init__()

        # The base __post_init__ might create a ViT-specific embedding layer.
        # We override it here with a deepcopy of the model's actual embedding layer.
        self.embedding_for_obfuscated_images = copy.deepcopy(self.original_embeddings)

    def _set_obfuscation(self, use_obfuscation: bool) -> None:
        if use_obfuscation:
            self.model.segformer.encoder.patch_embeddings = self.embedding_for_obfuscated_images
        else:
            self.model.segformer.encoder.patch_embeddings = self.original_embeddings

    def obfuscate(self, images: ImageInput, labels: Optional[Any] = None) -> ObfuscationOutput:
        if labels is not None:
            inputs = self.processor(images=images, segmentation_maps=labels, return_tensors='pt').to(self.model.device)
        else:
            inputs = self.processor(images=images, return_tensors='pt').to(self.model.device)
        
        obfuscated_images = self.obfuscator(inputs['pixel_values'])

        return ObfuscationOutput(
            obfuscated_images=obfuscated_images,
            processed_inputs=inputs
        )
    
    def forward_obfuscated_inputs(self, obfuscated_inputs: ObfuscationOutput) -> SegmentationModelOutputWithObfuscation:
        self._set_obfuscation(True)

        inputs = obfuscated_inputs.processed_inputs
        inputs['pixel_values'] = obfuscated_inputs.obfuscated_images

        model_outputs = self.model(**inputs)

        return SegmentationModelOutputWithObfuscation(
            obfuscated_images=obfuscated_inputs.obfuscated_images,
            model_outputs=model_outputs
        )
    
    def forward_clean_inputs(self, images: ImageInput, labels: Optional[Any] = None) -> SegmentationModelOutputWithObfuscation:
        self._set_obfuscation(False)

        if labels is not None:
            processed_inputs = self.processor(images=images, segmentation_maps=labels, return_tensors='pt').to(self.model.device)
        else:
            processed_inputs = self.processor(images=images, return_tensors='pt').to(self.model.device)

        model_outputs = self.model(**processed_inputs)

        return SegmentationModelOutputWithObfuscation(
            obfuscated_images=None,
            model_outputs=model_outputs
        )

    def forward(self, images: ImageInput, labels: Optional[Any] = None, with_obfuscation: bool = False) -> SegmentationModelOutputWithObfuscation:
        if with_obfuscation:
            obfuscated_output = self.obfuscate(images, labels)
            return self.forward_obfuscated_inputs(obfuscated_output)
        
        return self.forward_clean_inputs(images, labels)
    
    @staticmethod
    def train_model(
        model: nn.Module,
        dataset_config: SegmentationDatasetConfigWithTrainingAndEvaluationSplits,
        iterations: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 32,   
    ) -> nn.Module:
        train_dataset, _ = dataset_config.build()

        accelerator = accelerate.Accelerator()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=iterations)

        def collate_fn(batch):
            return {
                dataset_config.input_column: [item[dataset_config.input_column] for item in batch],
                dataset_config.label_column: [item[dataset_config.label_column] for item in batch]
            }

        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        model, optimizer, scheduler, dataloader = accelerator.prepare(
            model, optimizer, scheduler, dataloader
        )

        model.train()
        pbar = tqdm(itertools.islice(itertools.cycle(dataloader), iterations), total=iterations, desc='Training model')
        for batch in pbar:
            optimizer.zero_grad()

            outputs = model(
                images=batch[dataset_config.input_column], 
                labels=batch[dataset_config.label_column]
            )

            loss = outputs.model_outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({'loss': loss.item()})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        unwrapped_model = accelerator.unwrap_model(model).cpu()

        del model, optimizer, scheduler, dataloader
        return unwrapped_model
    
    @staticmethod
    def evaluate_model(
        model: nn.Module,
        dataset_config: SegmentationDatasetConfigWithTrainingAndEvaluationSplits,
        batch_size: int = 32,
        with_obfuscation: bool = False
    ) -> SegmentationEvaluationOutput:
        _, eval_dataset = dataset_config.build()

        def collate_fn(batch):
            images = [item[dataset_config.input_column] for item in batch]
            labels = [item[dataset_config.label_column] for item in batch]
            target_sizes = [img.size[::-1] for img in images] # (height, width)
            return images, labels, target_sizes

        accelerator = accelerate.Accelerator()

        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        model, dataloader = accelerator.prepare(model, dataloader)

        unwrapped_model = accelerator.unwrap_model(model)
        num_classes = len(dataset_config.id2label)
        metric = JaccardIndex(task='multiclass', num_classes=num_classes).to(accelerator.device)

        model.eval()
        for images, targets, target_sizes in tqdm(dataloader, desc='Evaluating model'):
            with torch.no_grad():
                outputs = model(images=images, with_obfuscation=with_obfuscation)

                predicted_masks = unwrapped_model.processor.post_process_semantic_segmentation(
                    outputs.model_outputs, target_sizes=target_sizes
                )
                predicted_masks = torch.stack(predicted_masks).to(accelerator.device)
                
                ground_truth_masks = torch.stack([
                    torch.from_numpy(np.array(t, dtype=np.int64)) for t in targets
                ]).to(accelerator.device)

                metric.update(predicted_masks, ground_truth_masks)

        computed_metrics = metric.compute()
        del model, dataloader, metric

        scalar_metrics = {'miou': computed_metrics.item()}
        return SegmentationEvaluationOutput(**scalar_metrics)


class ZeroShotImageSegmentationModel(BaseImageModelWithObfuscation):
    """
    Zero-shot image segmentation model based on CLIPSeg.
    """
    def __init__(
        self,
        config: ImageModelConfigWithObfuscation,
        pretrained: bool = True
      ) -> None:
        nn.Module.__init__(self)

        self.config = config

        if pretrained:
            self.model = CLIPSegForImageSegmentation.from_pretrained(config.hf_model_name_or_path)
        else:
            self.model = CLIPSegForImageSegmentation(config.hf_model_config)

        self.processor = CLIPSegProcessor.from_pretrained(config.hf_model_name_or_path)

        self.vision_config = self.config.hf_model_config.vision_config
        self.original_embeddings = self.model.clip.vision_model.embeddings

        self.__post_init__()

        self.embedding_for_obfuscated_images.cls_embedding.data = self.original_embeddings.class_embedding.data
        self.embedding_for_obfuscated_images.position_embedding.weight.data = self.original_embeddings.position_embedding.weight.data

    def _set_obfuscation(self, use_obfuscation: bool) -> None:
        if use_obfuscation:
            self.model.clip.vision_model.embeddings = self.embedding_for_obfuscated_images
        else:
            self.model.clip.vision_model.embeddings = self.original_embeddings

    def obfuscate(self, images: ImageInput, texts: Optional[List[List[str]]] = None) -> ObfuscationOutput:
        inputs = self.processor(images=images, texts=texts, padding=True, return_tensors='pt').to(self.model.device)
        obfuscated_images = self.obfuscator(inputs['pixel_values'])
        
        return ObfuscationOutput(obfuscated_images=obfuscated_images, processed_inputs=inputs)
    
    def forward_obfuscated_inputs(self, obfuscated_inputs: ObfuscationOutput) -> ModelOutputWithObfuscation:
        self._set_obfuscation(True)

        inputs = obfuscated_inputs.processed_inputs
        inputs['pixel_values'] = obfuscated_inputs.obfuscated_images

        return ModelOutputWithObfuscation(
            obfuscated_images=obfuscated_inputs.obfuscated_images,
            model_outputs=self.model(**inputs)
        )
    
    def forward_clean_inputs(self, images: ImageInput, texts: List[List[str]]) -> ModelOutputWithObfuscation:
        self._set_obfuscation(False)

        processed_inputs = self.processor(images=images, texts=texts, padding=True, return_tensors='pt').to(self.model.device)
        return ModelOutputWithObfuscation(obfuscated_images=None, model_outputs=self.model(**processed_inputs))
    
    def forward(self, images: ImageInput, texts: List[List[str]], with_obfuscation: bool = False) -> ModelOutputWithObfuscation:
        if with_obfuscation:
            return self.forward_obfuscated_inputs(self.obfuscate(images, texts))
        
        return self.forward_clean_inputs(images, texts)
    
    @staticmethod
    def evaluate_model(
        model: nn.Module,
        dataset_config: SegmentationDatasetConfigWithTrainingAndEvaluationSplits,
        batch_size: int = 32,
        with_obfuscation: bool = False
    ) -> SegmentationEvaluationOutput:
        _, eval_dataset = dataset_config.build()
        
        label_names = [name for _, name in sorted(dataset_config.id2label.items())]
        num_classes = len(label_names)

        def collate_fn(batch):
            images = [item[dataset_config.input_column] for item in batch]
            labels = [item[dataset_config.label_column] for item in batch]
            target_sizes = [img.size[::-1] for img in images]
            return images, labels, target_sizes

        accelerator = accelerate.Accelerator()

        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        model, dataloader = accelerator.prepare(model, dataloader)

        metric = JaccardIndex(task='multiclass', num_classes=num_classes).to(accelerator.device)

        model.eval()
        for images, targets, target_sizes in tqdm(dataloader, desc='Evaluating model'):
            with torch.no_grad():
                texts = [label_names] * len(images)
                outputs = model(images=images, texts=texts, with_obfuscation=with_obfuscation)

                logits = outputs.model_outputs.logits

                upsampled_logits_list = []
                for i in range(logits.shape[0]):
                    upsampled_logits = F.interpolate(
                        logits[i].unsqueeze(0),
                        size=target_sizes[i],
                        mode='bilinear',
                        align_corners=False
                    )
                    upsampled_logits_list.append(upsampled_logits.squeeze(0))

                predicted_masks = torch.stack([l.argmax(dim=0) for l in upsampled_logits_list])

                ground_truth_masks = torch.stack([
                    torch.from_numpy(np.array(t, dtype=np.int64)) for t in targets
                ]).to(accelerator.device)

                metric.update(predicted_masks, ground_truth_masks)

        computed_metrics = metric.compute()
        del model, dataloader, metric

        scalar_metrics = {'miou': computed_metrics.item()}
        return SegmentationEvaluationOutput(**scalar_metrics)