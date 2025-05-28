import os
import torch

from pathlib import Path
from tqdm import tqdm
from unsloth import FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig
from torch.utils.data import DataLoader
from typing import Optional, Dict

from unsloth.trainer import UnslothVisionDataCollator
from src.utils.logger import logger

class BaseTrainer:
    def __init__(
        self,
        trainer_config,
        model,
        tokenizer,
        train_dataset,
        val_dataset=None,
        is_save_model=True,
        data_collator=UnslothVisionDataCollator, 
        device="cuda",
    ):
        self.trainer_config = trainer_config
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.is_save_model = is_save_model
        self.data_collator = data_collator
        
        # Prepare model for Unsloth training
        FastVisionModel.for_training(self.model)

        self.sft_config = self._get_sft_config()
        
        # Initialize SFTTrainer
        self.trainer = self._get_sft_trainer()

    def _get_sft_trainer(self) -> SFTTrainer:
        """Create SFTTrainer instance with the provided configuration"""
        return SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            train_dataset = self.train_dataset,
            eval_dataset = self.val_dataset,
            dataset_text_field = "",  
            args = self.sft_config,
        )

    def _get_sft_config(self) -> SFTConfig:
        """Convert YAML config to SFTTrainer arguments"""
        
        return SFTConfig(
            **self.trainer_config
        )

    def train(self):
        """Execute the training workflow"""
        self.trainer.train()
        
        # Save final model and tokenizer
        if self.is_save_model:
            logger.info("Saving the trained model and tokenizer...")
            self._save_model()
        
    def _save_model(self):
        """Save trained model and tokenizer"""
        save_path = Path(self.sft_config.output_dir)
        self.model.save_pretrained(save_path / "final_model")
        self.tokenizer.save_pretrained(save_path / "final_model")
        logger.info(f"Model and tokenizer saved to {save_path}/final_model")

    def evaluate(self):
        """Run evaluation on validation dataset"""
        if self.trainer.eval_dataset is None:
            raise ValueError("No validation dataset provided for evaluation")
            
        metrics = self.trainer.evaluate()

        return metrics
    
    