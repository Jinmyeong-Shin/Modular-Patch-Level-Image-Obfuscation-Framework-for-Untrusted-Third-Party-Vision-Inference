from typing import Optional, Literal

import copy
import itertools

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import accelerate

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from data.type import ImageInput

from data.dataset_config import ImageClassificationDatasetConfigWithTrainingAndEvaluationSplits

from data.obfusccation_output import ObfuscationOutput
from data.model_output import ModelOutputWithObfuscation, ImageClassificationModelOutputWithObfuscation
from data.evaluation_output import ClassificationEvaluationOutput

from .base import BaseImageModelWithObfuscation

class ImageClassificationModel(BaseImageModelWithObfuscation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.classifier = nn.Linear(self.config.hf_model_config.hidden_size, self.config.num_classes)

        self.vision_config = self.config.hf_model_config

        self.original_embeddings = self.model.embeddings

        self.__post_init__()

        self.embedding_for_obfuscated_images.cls_embedding.data = self.original_embeddings.cls_token.data
        self.embedding_for_obfuscated_images.position_embedding = copy.deepcopy(self.original_embeddings.position_embeddings)

    def _set_obfuscation(self, use_obfuscation: bool) -> None:
        if use_obfuscation:
            self.model.embeddings = self.embedding_for_obfuscated_images
        else:
            self.model.embeddings = self.original_embeddings

    def obfuscate(self, images: ImageInput) -> ObfuscationOutput:
        inputs = self.processor(images=images, return_tensors='pt').to(self.model.device)
        obfuscated_images = self.obfuscator(inputs['pixel_values'])

        return ObfuscationOutput(
            obfuscated_images=obfuscated_images,
            processed_inputs=inputs
        )
    
    def forward_obfuscated_inputs(self, obfuscated_inputs: ObfuscationOutput) -> ImageClassificationModelOutputWithObfuscation:
        self._set_obfuscation(True)

        inputs = obfuscated_inputs.processed_inputs
        inputs['pixel_values'] = obfuscated_inputs.obfuscated_images

        vit_outputs = self.model(**inputs)

        logits = self.classifier(vit_outputs.pooler_output)

        return ImageClassificationModelOutputWithObfuscation(
            obfuscated_images=obfuscated_inputs.obfuscated_images,
            logits=logits,
            model_outputs=vit_outputs
        )
    
    def forward_clean_inputs(self, images: ImageInput) -> ImageClassificationModelOutputWithObfuscation:
        self._set_obfuscation(False)

        vit_outputs = self.model(**self.processor(images=images, return_tensors='pt').to(self.model.device))

        logits = self.classifier(vit_outputs.pooler_output)

        return ImageClassificationModelOutputWithObfuscation(
            obfuscated_images=None,
            logits=logits,
            model_outputs=vit_outputs
        )

    def forward(self, images: ImageInput, with_obfuscation: bool = False) -> ImageClassificationModelOutputWithObfuscation:
        if with_obfuscation:
            return self.forward_obfuscated_inputs(self.obfuscate(images))
        
        return self.forward_clean_inputs(images)
    
    @staticmethod
    def train_model(
        model: nn.Module,
        dataset_config: ImageClassificationDatasetConfigWithTrainingAndEvaluationSplits,
        iterations: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 256,   
    ) -> nn.Module:
        train_dataset, _ = dataset_config.build()

        accelerator = accelerate.Accelerator()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=iterations)

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        model, optimizer, scheduler, dataloader = accelerator.prepare(
            model, optimizer, scheduler, dataloader
        )

        model.train()
        pbar = tqdm(itertools.islice(itertools.cycle(dataloader), iterations), total=iterations, desc='Training model')
        for batch in pbar:
            optimizer.zero_grad()

            outputs = model(images=batch[dataset_config.input_column])

            loss = F.cross_entropy(outputs.logits, batch[dataset_config.label_column])

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({'loss': loss.item()})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        unwraped_model = accelerator.unwrap_model(model).cpu()

        del model, optimizer, scheduler, dataloader
        return unwraped_model
    
    @staticmethod
    def evaluate_model(
        model: nn.Module,
        dataset_config: ImageClassificationDatasetConfigWithTrainingAndEvaluationSplits,
        batch_size: int = 256,
        with_obfuscation: bool = False
    ) -> ClassificationEvaluationOutput:
        _, eval_dataset = dataset_config.build()

        accelerator = accelerate.Accelerator()

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        model, dataloader = accelerator.prepare(
            model, dataloader
        )

        logits = []
        labels = []

        model.eval()
        for batch in tqdm(dataloader, desc='Evaluating model'):
            with torch.no_grad():
                
                outputs = model(images=batch[dataset_config.input_column], with_obfuscation=with_obfuscation)

                logits.append(outputs.logits)
                labels.append(batch[dataset_config.label_column])

        logits = torch.cat(logits, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

        accuracy = accuracy_score(labels, np.argmax(logits, axis=1))
        recall = recall_score(labels, np.argmax(logits, axis=1), average='macro')
        precision = precision_score(labels, np.argmax(logits, axis=1), average='macro')
        f1 = f1_score(labels, np.argmax(logits, axis=1), average='macro')

        del model, dataloader

        return ClassificationEvaluationOutput(
            logits=logits,
            accuracy=accuracy,
            recall=recall,
            precision=precision,
            f1=f1
        )


class ZeroShotImageClassificationModel(BaseImageModelWithObfuscation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vision_config = self.config.hf_model_config.vision_config

        self.original_embeddings = self.model.vision_model.embeddings

        self.__post_init__()

        # Copy the class and position embeddings from the original pre-trained model.
        # This simplifies the deobfuscator's training task, as it only needs to learn
        # to reconstruct the patch content, not the boilerplate embeddings.
        self.embedding_for_obfuscated_images.cls_embedding.data = self.original_embeddings.class_embedding.data
        self.embedding_for_obfuscated_images.position_embedding.weight.data = self.original_embeddings.position_embedding.weight.data

    def _set_obfuscation(self, use_obfuscation: bool) -> None:
        if use_obfuscation:
            self.model.vision_model.embeddings = self.embedding_for_obfuscated_images
        else:
            self.model.vision_model.embeddings = self.original_embeddings

    def obfuscate(self, images: ImageInput, text: Optional[list[str]] = None) -> ObfuscationOutput:
        inputs = self.processor(images=images, text=text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        obfuscated_images = self.obfuscator(inputs['pixel_values'])
        
        return ObfuscationOutput(
            obfuscated_images=obfuscated_images,
            processed_inputs=inputs
        )
    
    def forward_obfuscated_inputs(self, obfuscated_inputs: ObfuscationOutput) -> ModelOutputWithObfuscation:
        self._set_obfuscation(True)

        inputs = obfuscated_inputs.processed_inputs
        inputs['pixel_values'] = obfuscated_inputs.obfuscated_images

        return ModelOutputWithObfuscation(
            obfuscated_images=obfuscated_inputs.obfuscated_images,
            model_outputs=self.model(**inputs)
        )
    
    def forward_clean_inputs(self, images: ImageInput, text: list[str]) -> ModelOutputWithObfuscation:
        self._set_obfuscation(False)

        return ModelOutputWithObfuscation(
            obfuscated_images=None,
            model_outputs=self.model(**self.processor(images=images, text=text, padding=True, truncation=True, return_tensors='pt').to(self.model.device))
        )
    
    def forward(self, images: ImageInput, text: list[str], with_obfuscation: bool = False) -> ModelOutputWithObfuscation:
        if with_obfuscation:
            return self.forward_obfuscated_inputs(self.obfuscate(images, text))
        
        return self.forward_clean_inputs(images, text)
    
    @staticmethod
    def evaluate_model(
        model: nn.Module,
        dataset_config: ImageClassificationDatasetConfigWithTrainingAndEvaluationSplits,
        batch_size: int = 256,
        with_obfuscation: bool = False
    ) -> ClassificationEvaluationOutput:
        _, eval_dataset = dataset_config.build()
        
        label_names = [name for cls, name in dataset_config.id2label.items()]

        accelerator = accelerate.Accelerator()

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        model, dataloader = accelerator.prepare(
            model, dataloader
        )

        logits = []
        labels = []

        model.eval()
        for batch in tqdm(dataloader, desc='Evaluating model'):
            with torch.no_grad():
                
                outputs = model(images=batch[dataset_config.input_column], text=label_names, with_obfuscation=with_obfuscation)

                logits.append(outputs.model_outputs.logits_per_image)
                labels.append(batch[dataset_config.label_column])

        logits = torch.cat(logits, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

        accuracy = accuracy_score(labels, np.argmax(logits, axis=1))
        recall = recall_score(labels, np.argmax(logits, axis=1), average='macro')
        precision = precision_score(labels, np.argmax(logits, axis=1), average='macro')
        f1 = f1_score(labels, np.argmax(logits, axis=1), average='macro')

        del model, dataloader

        return ClassificationEvaluationOutput(
            logits=logits,
            accuracy=accuracy,
            recall=recall,
            precision=precision,
            f1=f1
        )
                

