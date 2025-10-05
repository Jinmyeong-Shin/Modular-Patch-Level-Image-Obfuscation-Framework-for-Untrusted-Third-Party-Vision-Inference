from abc import ABC, abstractmethod
from typing import TypeVar

import itertools

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoProcessor

import accelerate

import datasets

from data.model_config import ImageModelConfigWithObfuscation
from data.dataset_config import DatasetConfig
from data.evaluation_output import EvaluationOutput

from network.obfuscation import ChannelWisePatchLevelObfuscator
from network.embedding import ObfuscationEmbedding

ImageModelConfigWithObfuscation_T = TypeVar('ImageModelConfigWithObfuscation_T', bound=ImageModelConfigWithObfuscation)
DatasetConfig_T = TypeVar('DatasetConfig_T', bound=DatasetConfig)
EvaluationOutput_T = TypeVar('EvaluationOutput_T', bound=EvaluationOutput)

class BaseImageModelWithObfuscation(
    ABC,
    nn.Module
):
    def __init__(
        self,
        config: ImageModelConfigWithObfuscation_T,
        pretrained: bool = True
    ) -> None:
        super().__init__()

        # Base model
        self.config = config

        if pretrained:
            self.model = AutoModel.from_pretrained(config.hf_model_name_or_path)
        else:
            self.model = AutoModel(config.hf_model_config)

        self.processor = AutoProcessor.from_pretrained(config.hf_model_name_or_path)

        self.vision_config = None
        self.original_embeddings = None

    def __post_init__(self):
        assert self.vision_config is not None, 'vision_config must be set for models with obfuscation'
        assert self.original_embeddings is not None, 'original_embeddings must be set for models with obfuscation'

        # Obfuscation modules
        self.obfuscator = ChannelWisePatchLevelObfuscator(
            image_size=self.vision_config.image_size, 
            num_channels=self.vision_config.num_channels, 
            patch_size=self.config.obfuscation_patch_size, 
            group_size=self.config.obfuscation_group_size        
        )
        self.embedding_for_obfuscated_images = ObfuscationEmbedding(
            image_size=self.vision_config.image_size,
            num_channels=self.vision_config.num_channels,
            patch_size=self.vision_config.patch_size,
            embed_dim=self.vision_config.hidden_size
        )

    @abstractmethod
    def _set_obfuscation(self, use_obfuscation: bool) -> None:
        pass

    @staticmethod
    def train_obfuscation_modules(
        model: nn.Module,
        iterations: int,
        learning_rate: float = 1e-2,
        batch_size: int = 32
    ) -> nn.Module:
        def _transform_fn(example):
            if isinstance(example['image'], list):
                example['image'] = [img.convert('RGB') for img in example['image']]
            else:
                example['image'] = example['image'].convert('RGB')
            return example

        dataset = datasets.load_dataset('benjamin-paine/imagenet-1k-256x256', split='train')
        dataset.set_transform(_transform_fn)
        dataset.set_format('torch')
        
        accelerator = accelerate.Accelerator()
        
        optimizer = torch.optim.AdamW(model.embedding_for_obfuscated_images.patch_embedding.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=iterations)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, dataloader
        )

        model.train()
        pbar = tqdm(itertools.islice(itertools.cycle(dataloader), iterations), total=iterations, desc='Training embedding_for_obfuscated_images layer')
        for batch in pbar:
            optimizer.zero_grad()

            obfuscated = model.obfuscate(images=batch['image'])

            original_embeddings = model.original_embeddings(obfuscated.processed_inputs['pixel_values'])
            obfuscated_embeddings = model.embedding_for_obfuscated_images(obfuscated.obfuscated_images)

            loss = F.mse_loss(obfuscated_embeddings, original_embeddings.detach())

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            pbar.set_postfix({'loss': loss.item(), 'lr': lr_scheduler.get_last_lr()[0]})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        unwraped_model = accelerator.unwrap_model(model).cpu()

        del model, optimizer, lr_scheduler, dataloader
        return unwraped_model

    @staticmethod
    @abstractmethod
    def evaluate_model(
        model: nn.Module,
        dataset_config: DatasetConfig_T,
        batch_size: int = 32,
        with_obfuscation: bool = False
    ) -> EvaluationOutput_T:
        pass

            
