import torch

from unsloth import FastVisionModel

class GetBaseVLM:
    """
    A class to load a base model from the Hugging Face model hub.
    
    Attributes:
        model_name (str): The name of the model to load.
        pretrained (bool): Whether to load the pretrained weights.
    """
    def __init__(self, model_name, lora_config, pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained
        self.lora_config = lora_config


    def load_base_model(self):
        """
        Load the base model from the Hugging Face model hub.
        
        Returns:
            torch.nn.Module: The loaded model.
        """
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_name,
            load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        )

        return self.model, self.tokenizer

    def load_lora_model(self):
        """
        Load the LoRA model.
        
        Returns:
            torch.nn.Module: The loaded LoRA model.
        """
        self.model = FastVisionModel.get_peft_model(self.model,
            finetune_vision_layers = False, 
            finetune_language_layers = False,
            finetune_attention_modules = True, 
            finetune_mlp_modules = True, 
            r = 16,         
            lora_alpha = 16,  
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  
            loftq_config = None, 
            # target_modules = "all-linear", 
        )

