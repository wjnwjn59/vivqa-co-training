from tqdm import tqdm
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
from typing import Dict, Any
from pathlib import Path
import argparse
import logging
# Import functions from utils
from utils.utils import (
    set_seed, read_system_prompt, read_user_prompt, extract_paraphrase,
    process_with_jinja_template, save_results_to_json, load_model, setup_logging
)

# Configure logging
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(question: str, image_path: str, system_instruction: str, user_prompt: str) -> str:
    """
    Perform inference on an image-question pair using the Qwen2.5-VL model.
    """
    global model, processor
    
    try:
        # Prepare messages for the system and the user
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        
        # Generate text prompt using the processor's chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision information from messages
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs for the model
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # Generate output tokens from the model
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the output tokens and extract the answer
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        raw_response = output_text[0]
        # Extract the structured information
        paraphrased_question = extract_paraphrase(raw_response)
        
        return paraphrased_question if paraphrased_question else raw_response
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return f"Error processing question: {question}"

def read_and_process_data(json_path: str, image_folder: str, system_instruction: str, user_prompt_template: str) -> Dict[str, Any]:
    """
    Read JSON data with question IDs as keys, process each entry, and
    generate paraphrased questions using the model.
    """
    try:
        # Convert paths to Path objects
        json_path = Path(json_path)
        image_folder = Path(image_folder)
        
        # Read the JSON data
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create a dictionary to store results
        results: Dict[str, Any] = {}
        
        # Process each entry in the data
        for question_id, entry in tqdm(data.items(), desc="Processing questions"):
            try:
                # Extract fields
                image_id = entry.get("image_id")
                question = entry.get("question")
                answer = entry.get("answer")
                
                # Create the path to the image
                image_path = image_folder / f"{int(image_id):012d}.jpg"
                
                # Generate user prompt from template
                user_prompt = process_with_jinja_template(
                    question_id, question, user_prompt_template
                )
                
                # Get raw response from model
                raw_response = inference(question, str(image_path), system_instruction, user_prompt)
                
                # Extract structured information
                paraphrased_question = extract_paraphrase(raw_response)
                
                # Use extracted or raw response
                paraphrase = paraphrased_question if paraphrased_question else raw_response
                
                # Store result with original data plus generated content
                results[question_id] = {
                    "image_id": image_id,
                    "question": question,
                    "answer": answer,
                    "qwenvl_paraphrased": {
                        "question_paraphrased": paraphrase
                    }
                }
            except Exception as e:
                logger.error(f"Error processing question ID {question_id}: {e}")
                continue
                
        return results
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return {}

def main() -> None:
    """Main function to run the Qwen2.5-VL paraphrasing pipeline."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Qwen2.5-VL Inference for Question Paraphrasing')
    
    # Required arguments
    parser.add_argument('--system_prompt', type=str, required=True,
                        help='Path to system prompt file')
    parser.add_argument('--user_prompt', type=str, required=True,
                        help='Path to user prompt template file')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to folder containing images')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Path to input JSON file with questions')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Path to output JSON file for results')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_path', type=str, default='logs/qwenvl.log',
                        help='Path to log file (default: logs/qwenvl.log)')
    
    args = parser.parse_args()
    
    # Setup logging with the specified path
    setup_logging(args.log_path)
    
    # Set global device based on args if needed
    global device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Set random seed
    set_seed(args.seed)
    
    try:
        # Load model and processor using hardcoded values
        global model, processor
        model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
        cache_dir = '../../weight/vlm/qwen2.5-vl-3b-instruct'
        model, processor = load_model(model_name, cache_dir)
        
        # Read prompt files
        system_instruction = read_system_prompt(args.system_prompt)
        user_prompt_template = read_user_prompt(args.user_prompt)
        
        if not system_instruction:
            logger.warning("System prompt is empty. Using default instruction.")
            system_instruction = "You are a helpful assistant that generates paraphrases of questions."
        
        if not user_prompt_template:
            logger.warning("User prompt template is empty. Using default template.")
            user_prompt_template = """
            {% for pair in qa_pairs %}
            Question ID: {{ pair.questionId }}  
            Original Question: {{ pair.question }}  
            {% endfor %}
            """
        
        # Process data
        results = read_and_process_data(
            args.input_json, 
            args.image_folder,
            system_instruction,
            user_prompt_template
        )
        
        # Save results
        save_results_to_json(results, args.output_json)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
