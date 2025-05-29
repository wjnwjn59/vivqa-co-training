import torch
import os

from unsloth import FastVisionModel
from transformers import TextStreamer


from src.utils.configurate import load_yaml_config
from src.utils.logger import logger
from src.prompts.prompt_formatters import convert_to_inference_conversation
from src.models.base_model import load_vlm
from src.utils.img_handler import load_image

def predict(model, tokenizer, input_text, input_image, generation_config={}, device="cuda"):
    inputs = tokenizer(input_image,
                       input_text,
                       add_special_tokens=False,
                       return_tensors="pt").to(device)
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    predictions = model.generate(**inputs, 
                                 streamer=text_streamer, 
                                 **generation_config)
    return predictions

def run_inference(model, tokenizer, question, ):
    pass

if __name__ == "__main__":
    # Load configuration files
    config = load_yaml_config()
    logger.info("Configuration:%s", config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    model, tokenizer = load_vlm(
        model_name=config["vqa_model"]["name"],
        lora_config=None,
        quantize_config=config["vqa_model"].get("quantize_config", None),
        dtype=config["vqa_model"].get("dtype", "float16"),
        device=config["vqa_model"].get("device", "cuda")
    )
    FastVisionModel.for_inference(model)
    
    question = "What is the animal in the image?"
    image = "static/cat.jpg"  
    pil_image = load_image(image)
    
    messages = convert_to_inference_conversation(question, template_name=config["vqa_model"]["instruct_template"])
    print(messages)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(input_text)
    
    predictions = predict(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        input_image=pil_image,
        generation_config=config["vqa_model"]["generation_config"],
        device=device
    )


    print("Predictions:", predictions)