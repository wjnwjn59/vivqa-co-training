import torch
from unsloth import FastVisionModel

dtype_mapping = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def apply_lora(model, lora_config: dict):
    """Apply LoRA configuration to model"""
    return FastVisionModel.get_peft_model(
        model,
        **lora_config
    )

def load_vlm(model_name, lora_config, quantize_config=None, dtype="float16", device="cuda"):
    """
    Load a vision-language model and tokenizer with optional LoRA
    """

    # Load base model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        use_gradient_checkpointing="unsloth",
        dtype=dtype_mapping[dtype],
        **quantize_config if quantize_config is not None else {}
    )
    
    # Apply LoRA if config provided
    if lora_config:
        model = apply_lora(model, lora_config)
    
    return model, tokenizer