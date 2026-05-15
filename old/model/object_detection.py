from typing import Optional, Any, List
from dataclasses import dataclass

import copy
import itertools
import json

from tqdm import tqdm

import torch
import torch.nn as nn

import accelerate
try:
    # For torchmetrics >= v0.11.0
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import AutoModelForObjectDetection, AutoProcessor, OwlViTForObjectDetection

from data.type import ImageInput

# The following are placeholders for data types that should exist in your project.
# You would need to create/define these in the respective `data` module files.
# For example, `ObjectDetectionDatasetConfigWithTrainingAndEvaluationSplits` would be
# the object detection equivalent of the classification one.
ObjectDetectionDatasetConfigWithTrainingAndEvaluationSplits = Any

from data.model_config import ImageModelConfigWithObfuscation

from data.obfusccation_output import ObfuscationOutput
from data.model_output import ModelOutputWithObfuscation, ObjectDetectionModelOutputWithObfuscation
from data.evaluation_output import ObjectDetectionEvaluationOutput

from .base import BaseImageModelWithObfuscation


class ObjectDetectionModel(BaseImageModelWithObfuscation):
    """
    Object detection model based on a fine-tunable Vision Transformer (ViT) architecture like YOLOS.
    """
    def __init__(
        self, 
        config: ImageModelConfigWithObfuscation,
        pretrained: bool = True
    ) -> None:
        nn.Module.__init__(self)

        self.config = config

        if pretrained:
            self.model = AutoModelForObjectDetection.from_pretrained(config.hf_model_name_or_path)
        else:
            self.model = AutoModelForObjectDetection(config.hf_model_config)

        self.processor = AutoProcessor.from_pretrained(config.hf_model_name_or_path)

        self.vision_config = self.model.config

        self.original_embeddings = self.model.vit.embeddings

        self.__post_init__()

        self.embedding_for_obfuscated_images.cls_embedding.data = self.original_embeddings.cls_token.data
        self.embedding_for_obfuscated_images.detection_tokens = copy.deepcopy(self.original_embeddings.detection_tokens)
        self.embedding_for_obfuscated_images.position_embedding = copy.deepcopy(self.original_embeddings.position_embeddings)

    def _set_obfuscation(self, use_obfuscation: bool) -> None:
        if use_obfuscation:
            self.model.vit.embeddings = self.embedding_for_obfuscated_images
        else:
            self.model.vit.embeddings = self.original_embeddings

    def obfuscate(self, images: ImageInput, labels: Optional[Any] = None) -> ObfuscationOutput:
        if labels is not None:
            inputs = self.processor(images=images, annotations=labels, size={'height': 768, 'width': 768}, return_tensors='pt').to(self.model.device)
        else:
            inputs = self.processor(images=images, size={'height': 768, 'width': 768}, return_tensors='pt').to(self.model.device)
        
        obfuscated_images = self.obfuscator(inputs['pixel_values'])

        return ObfuscationOutput(
            obfuscated_images=obfuscated_images,
            processed_inputs=inputs
        )
    
    def forward_obfuscated_inputs(self, obfuscated_inputs: ObfuscationOutput) -> ObjectDetectionModelOutputWithObfuscation:
        self._set_obfuscation(True)

        inputs = obfuscated_inputs.processed_inputs
        inputs['pixel_values'] = obfuscated_inputs.obfuscated_images

        model_outputs = self.model(**inputs)

        return ObjectDetectionModelOutputWithObfuscation(
            obfuscated_images=obfuscated_inputs.obfuscated_images,
            model_outputs=model_outputs
        )
    
    def forward_clean_inputs(self, images: ImageInput, labels: Optional[Any] = None) -> ObjectDetectionModelOutputWithObfuscation:
        self._set_obfuscation(False)

        if labels is not None:
            processed_inputs = self.processor(images=images, annotations=labels, size={'height': 768, 'width': 768}, return_tensors='pt').to(self.model.device)
        else:
            processed_inputs = self.processor(images=images, size={'height': 768, 'width': 768}, return_tensors='pt').to(self.model.device)

        model_outputs = self.model(**processed_inputs)

        return ObjectDetectionModelOutputWithObfuscation(
            obfuscated_images=None,
            model_outputs=model_outputs
        )

    def forward(self, images: ImageInput, labels: Optional[Any] = None, with_obfuscation: bool = False) -> ObjectDetectionModelOutputWithObfuscation:
        if with_obfuscation:
            obfuscated_output = self.obfuscate(images, labels)
            return self.forward_obfuscated_inputs(obfuscated_output)
        
        return self.forward_clean_inputs(images, labels)
    
    @staticmethod
    def train_model(
        model: nn.Module,
        dataset_config: ObjectDetectionDatasetConfigWithTrainingAndEvaluationSplits,
        iterations: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 32,   
    ) -> nn.Module:
        train_dataset, _ = dataset_config.build()

        accelerator = accelerate.Accelerator()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=iterations)

        def collate_fn(batch):
            processed_labels = []
            for item in batch:
                label_data = item[dataset_config.label_column]
                if isinstance(label_data, str):
                    try:
                        label_data = json.loads(label_data)
                    except json.JSONDecodeError:
                        raise ValueError(f"Label data for an item is a string but not valid JSON: {label_data}")
                processed_labels.append(label_data)
            return {
                dataset_config.input_column: [item[dataset_config.input_column] for item in batch],
                dataset_config.label_column: processed_labels
            }

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
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
        dataset_config: ObjectDetectionDatasetConfigWithTrainingAndEvaluationSplits,
        batch_size: int = 32,
        with_obfuscation: bool = False
    ) -> ObjectDetectionEvaluationOutput:
        _, eval_dataset = dataset_config.build()

        def collate_fn(batch):
            processed_labels = []
            for item in batch:
                label_data = item[dataset_config.label_column]
                if isinstance(label_data, str):
                    try:
                        label_data = json.loads(label_data)
                    except json.JSONDecodeError:
                        raise ValueError(f"Label data for an item is a string but not valid JSON: {label_data}")
                processed_labels.append(label_data)
            images = [item[dataset_config.input_column] for item in batch]
            target_sizes = torch.tensor([img.size[::-1] for img in images]) # (height, width)
            return images, processed_labels, target_sizes

        accelerator = accelerate.Accelerator()

        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        model, dataloader = accelerator.prepare(model, dataloader)

        unwrapped_model = accelerator.unwrap_model(model)
        metric = MeanAveragePrecision(box_format='xyxy', max_detection_thresholds=[1, 10, 100]).to(accelerator.device)

        model.eval()
        for images, targets, target_sizes in tqdm(dataloader, desc='Evaluating model'):
            with torch.no_grad():
                outputs = model(images=images, with_obfuscation=with_obfuscation)

                orig_target_sizes = target_sizes.to(accelerator.device)
                results = unwrapped_model.processor.post_process_object_detection(outputs.model_outputs, target_sizes=orig_target_sizes, threshold=0.1)

                preds = [{'boxes': res['boxes'], 'scores': res['scores'], 'labels': res['labels']} for res in results]

                formatted_targets = []
                for target_annos in targets:
                    boxes = []
                    labels = []

                    for anno in target_annos:
                        box_coco = anno['bbox'] # [x, y, w, h]
                        # convert to xyxy for torchmetrics
                        box_xyxy = [box_coco[0], box_coco[1], box_coco[0] + box_coco[2], box_coco[1] + box_coco[3]]
                        boxes.append(box_xyxy)
                        labels.append(anno['category_id'])
                    
                    formatted_targets.append({
                        'boxes': torch.tensor(boxes, dtype=torch.float32).to(accelerator.device),
                        'labels': torch.tensor(labels, dtype=torch.int64).to(accelerator.device)
                    })
                metric.update(preds, formatted_targets)

        computed_metrics = metric.compute()
        del model, dataloader, metric

        # Filter for scalar metrics, excluding per-class metrics and the 'classes' tensor
        # which might have a single element if there's only one class.
        scalar_metrics = {k: v.item() for k, v in computed_metrics.items()
                          if v.numel() == 1 and not k.endswith('_per_class') and k != 'classes'}
        return ObjectDetectionEvaluationOutput(**scalar_metrics)


class ZeroShotObjectDetectionModel(BaseImageModelWithObfuscation):
    """
    Zero-shot object detection model based on OWL-ViT.
    """
    def __init__(
        self,
        config: ImageModelConfigWithObfuscation,
        pretrained: bool = True
      ) -> None:
        nn.Module.__init__(self)

        self.config = config

        if pretrained:
            self.model = OwlViTForObjectDetection.from_pretrained(config.hf_model_name_or_path)
        else:
            self.model = OwlViTForObjectDetection(config.hf_model_config)

        self.processor = AutoProcessor.from_pretrained(config.hf_model_name_or_path)

        self.vision_config = self.config.hf_model_config.vision_config
        self.original_embeddings = self.model.owlvit.vision_model.embeddings

        self.__post_init__()

        self.embedding_for_obfuscated_images.cls_embedding.data = self.original_embeddings.class_embedding.data
        self.embedding_for_obfuscated_images.position_embedding.weight.data = self.original_embeddings.position_embedding.weight.data

    def _set_obfuscation(self, use_obfuscation: bool) -> None:
        if use_obfuscation:
            self.model.owlvit.vision_model.embeddings = self.embedding_for_obfuscated_images
        else:
            self.model.owlvit.vision_model.embeddings = self.original_embeddings

    def obfuscate(self, images: ImageInput, text: Optional[List[List[str]]] = None) -> ObfuscationOutput:
        inputs = self.processor(images=images, text=text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
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
    
    def forward_clean_inputs(self, images: ImageInput, text: List[List[str]]) -> ModelOutputWithObfuscation:
        self._set_obfuscation(False)

        processed_inputs = self.processor(images=images, text=text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        return ModelOutputWithObfuscation(obfuscated_images=None, model_outputs=self.model(**processed_inputs))
    
    def forward(self, images: ImageInput, text: List[List[str]], with_obfuscation: bool = False) -> ModelOutputWithObfuscation:
        if with_obfuscation:
            return self.forward_obfuscated_inputs(self.obfuscate(images, text))
        
        return self.forward_clean_inputs(images, text)
    
    @staticmethod
    def evaluate_model(
        model: nn.Module,
        dataset_config: ObjectDetectionDatasetConfigWithTrainingAndEvaluationSplits,
        batch_size: int = 32,
        with_obfuscation: bool = False
    ) -> ObjectDetectionEvaluationOutput:
        _, eval_dataset = dataset_config.build()
        
        label_names = [name for _, name in sorted(dataset_config.id2label.items())]

        def collate_fn(batch):
            processed_labels = []
            for item in batch:
                label_data = item[dataset_config.label_column]
                if isinstance(label_data, str):
                    try:
                        label_data = json.loads(label_data)
                    except json.JSONDecodeError:
                        raise ValueError(f"Label data for an item is a string but not valid JSON: {label_data}")
                processed_labels.append(label_data)
            images = [item[dataset_config.input_column] for item in batch]
            target_sizes = torch.tensor([img.size[::-1] for img in images])
            return images, processed_labels, target_sizes

        accelerator = accelerate.Accelerator()

        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        model, dataloader = accelerator.prepare(model, dataloader)

        unwrapped_model = accelerator.unwrap_model(model)
        metric = MeanAveragePrecision(box_format='xyxy', max_detection_thresholds=[1, 10, 100]).to(accelerator.device)

        model.eval()
        for images, targets, target_sizes in tqdm(dataloader, desc='Evaluating model'):
            with torch.no_grad():
                text_queries = [label_names] * len(images)
                outputs = model(images=images, text=text_queries, with_obfuscation=with_obfuscation)

                orig_target_sizes = target_sizes.to(accelerator.device)
                results = unwrapped_model.processor.post_process_grounded_object_detection(outputs.model_outputs, target_sizes=orig_target_sizes, threshold=0.1)

                preds = [{'boxes': res['boxes'], 'scores': res['scores'], 'labels': res['labels']} for res in results]
                
                formatted_targets = []
                for target_annos in targets:
                    boxes = []
                    labels = []
                    for anno in target_annos:
                        box_coco = anno['bbox'] # [x, y, w, h]
                        # convert to xyxy for torchmetrics
                        box_xyxy = [box_coco[0], box_coco[1], box_coco[0] + box_coco[2], box_coco[1] + box_coco[3]]
                        boxes.append(box_xyxy)
                        labels.append(anno['category_id'])
                    
                    formatted_targets.append({
                        'boxes': torch.tensor(boxes, dtype=torch.float32).to(accelerator.device),
                        'labels': torch.tensor(labels, dtype=torch.int64).to(accelerator.device)
                    })
                metric.update(preds, formatted_targets)

        computed_metrics = metric.compute()
        del model, dataloader, metric

        # Filter for scalar metrics, excluding per-class metrics and the 'classes' tensor
        # which might have a single element if there's only one class.
        scalar_metrics = {k: v.item() for k, v in computed_metrics.items()
                          if v.numel() == 1 and not k.endswith('_per_class') and k != 'classes'}
        return ObjectDetectionEvaluationOutput(**scalar_metrics)
