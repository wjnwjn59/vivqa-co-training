import argparse
import torch

from src.data import *
from src.utils.configurate import load_yaml_config
from src.models.base_model import load_vlm
from src.training.trainers.base_trainer import BaseTrainer

from unsloth.trainer import UnslothVisionDataCollator
from src.prompts.prompt_formatters import convert_to_training_conversation

from src.utils.logger import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Co-training for Vietnamese VQA")
    parser.add_argument("--dataset", type=str, default="openvivqa", 
                      choices=["openvivqa", "vitextvqa"],
                      help="Dataset name (default: openvivqa)")
    parser.add_argument("--root_dir", type=str, default="/mnt/VLAI_data/OpenViVQA",
                      help="Root directory for the dataset")
    parser.add_argument("--training_type", type=str, default="base",
                        choices=["base", "cotraining"],
                        help="Training type (default: base)")
    parser.add_argument("--is_eval", action="store_true",
                      help="Whether to evaluate the model while training")
    parser.add_argument("--data_collator_type", type=str, default="unsloth",
                        choices=["custom", "unsloth"],
                        help="Type of data collator to use (default: custom)")
    return parser.parse_args()


def select_dataset(args):
    if args.dataset == "openvivqa":
        train_dataset = OpenViVQADataset(root_dir=args.root_dir, subset="train")
        val_dataset = OpenViVQADataset(root_dir=args.root_dir, subset="dev") if args.is_eval else None
    elif args.dataset == "vitextvqa":  
        train_dataset = ViTextVQADataset(root_dir=args.root_dir, subset="train")
        val_dataset = ViTextVQADataset(root_dir=args.root_dir, subset="dev") if args.is_eval else None  
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
    
    return train_dataset, val_dataset

def get_data_collator(model, tokenizer, data_collator_type="unsloth", template_name=""):
    """Initialize Unsloth-specific data collator"""
    if data_collator_type == "custom":
        return CustomVQADataCollator(model=model, 
                                     processor=tokenizer,
                                     formatting_func=lambda x: convert_to_training_conversation(x, template_name=template_name))
    elif data_collator_type == "unsloth":
        return UnslothVisionDataCollator(model=model, 
                                         processor=tokenizer,
                                         formatting_func=lambda x: convert_to_training_conversation(x, template_name=template_name))
    else:
        raise ValueError(f"Data collator {data_collator_type} is not supported. Use 'unsloth' or 'custom'.")

if __name__ == "__main__":
    args = parse_args() 
    # Load configuration
    config = load_yaml_config("configs")
    # validate_config(config)
    logger.info("Configuration:\n%s", config)
    logger.info("Arguments:\n%s", args)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    # Load model and tokenizer
    if config["vqa_model"]["quantization"]["is_quantize"]:
        quantize_config = config["vqa_model"]["quantization"].get("quantize_config", None)
    else:
        quantize_config = None

    logger.info("Load model...")
    model, tokenizer = load_vlm(
        model_name=config["vqa_model"]["name"],
        lora_config=config["lora"]["lora_config"] if config["lora"]["is_lora"] else None,
        quantize_config=quantize_config,
        dtype=config["vqa_model"]["dtype"],
        device=device
    )

    # Initialize dataset and dataloader
    logger.info("Load dataset...")
    train_dataset, val_dataset = select_dataset(args)
    logger.info(f"Train dataset size: {len(train_dataset)}")

    # Train the model
    logger.info("Starting training...")
    data_collator = get_data_collator(
        model=model,
        tokenizer=tokenizer,
        data_collator_type=args.data_collator_type,
        template_name=config["vqa_model"]["instruct_template"]
    )
    trainer = BaseTrainer(
        trainer_config=config["training"]["trainer_config"],
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        is_save_model=config["training"]["is_save_model"],
        data_collator=data_collator,
        device=device
    )
    trainer.train()
    logger.info("Training complete.")