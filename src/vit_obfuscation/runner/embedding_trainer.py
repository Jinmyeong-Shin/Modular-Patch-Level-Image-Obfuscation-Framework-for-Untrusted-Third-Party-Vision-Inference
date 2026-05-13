from __future__ import annotations

import itertools

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import EmbeddingTrainingConfig
from ..datasets.loader import load_embedding_training_dataset
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator


class EmbeddingTrainer:
    """
    Task-agnostic trainer for the ObfuscationEmbedding.

    Trains the patch embedding to minimize MSE between embeddings produced
    from obfuscated images and embeddings from clean images via the original
    model embedding layer.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: EmbeddingTrainingConfig,
    ) -> None:
        self.adapter = adapter
        self.obfuscator = obfuscator
        self.obf_embedding = obf_embedding
        self.processor = processor
        self.config = config

    def train(self) -> None:
        """Run embedding training loop."""
        dataset = load_embedding_training_dataset(
            self.config.training_dataset,
        )

        accelerator = accelerate.Accelerator()

        # Only train the patch embedding parameters
        optimizer = torch.optim.AdamW(
            self.obf_embedding.patch_embedding.parameters(),
            lr=self.config.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.iterations,
        )

        def collate_fn(batch):
            return {"image": [item["image"] for item in batch]}

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Wrap model components for accelerator
        model = self.adapter.model
        obfuscator = self.obfuscator
        obf_embedding = self.obf_embedding
        original_embeddings = self.adapter.original_embeddings

        model, obfuscator, obf_embedding, optimizer, lr_scheduler, dataloader = (
            accelerator.prepare(
                model, obfuscator, obf_embedding, optimizer, lr_scheduler, dataloader
            )
        )

        # Get the original embeddings from the (possibly wrapped) model
        unwrapped_model = accelerator.unwrap_model(model)
        adapter_for_original = ModelAdapter(unwrapped_model, self.adapter.spec)
        original_embeddings = adapter_for_original.original_embeddings

        obf_embedding.train()
        pbar = tqdm(
            itertools.islice(itertools.cycle(dataloader), self.config.iterations),
            total=self.config.iterations,
            desc="Training obfuscation embedding",
        )

        for batch in pbar:
            optimizer.zero_grad()

            # Process images
            images = batch["image"]
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(accelerator.device)

            # Get clean embeddings from original model
            with torch.no_grad():
                clean_embeddings = original_embeddings(pixel_values)
                # Some models (e.g., BEiT) return a tuple; take the first element
                if isinstance(clean_embeddings, tuple):
                    clean_embeddings = clean_embeddings[0]

            # Get obfuscated embeddings
            obfuscated_pixels = obfuscator(pixel_values)
            obfuscated_embeddings = obf_embedding(obfuscated_pixels)

            loss = F.mse_loss(obfuscated_embeddings, clean_embeddings.detach())

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Sync back to CPU
        self.obf_embedding = accelerator.unwrap_model(obf_embedding).cpu()
        self.obfuscator = accelerator.unwrap_model(obfuscator).cpu()
        self.adapter.model = accelerator.unwrap_model(model).cpu()
