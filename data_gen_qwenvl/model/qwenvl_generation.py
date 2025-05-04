from tqdm import tqdm
import json
import os
import torch
from typing import Dict, Any, List
from pathlib import Path
import argparse
import logging
import yaml
from qwen_vl_utils import process_vision_info

# Fix incorrect import path
from src.utils import (
    set_seed, read_system_prompt, read_user_prompt, format_output_json,
    process_nested_questions_with_template, save_results_to_json, load_model, setup_logging
)

# Define global variables at module level
model = None
processor = None

# Configure logging
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(image_path: str, system_instruction: str, user_prompt: str) -> str:
    """
    Perform inference with optimized performance.
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
        
        # Use inference_mode for better performance than no_grad
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False  # Deterministic generation is faster
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the output tokens
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output_text[0]
        
    except Exception as e:
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"Error during inference: {e}")
        return f"Error during inference: {str(e)}"

def read_and_process_data(
    json_path: str,
    image_folder: str,
    system_instruction: str,
    user_prompt_template: str,
    output_json_path: str
) -> Dict[str, Any]:
    """
    Read JSON data with nested question structure, process each entry, and
    generate alternative questions using the model.
    Periodically saves partial results to the output JSON after every 10 images.
    """
    try:
        # Try using faster JSON library if available
        try:
            import ujson as json_lib
        except ImportError:
            import json as json_lib
            
        # Convert paths to Path objects
        json_path = Path(json_path)
        image_folder = Path(image_folder)
        
        # Read the JSON data
        with json_path.open("r", encoding="utf-8") as f:
            data = json_lib.load(f)
        
        # Process nested questions and prepare for inference
        processing_info = process_nested_questions_with_template(data, user_prompt_template, str(image_folder))
        
        # Create a dictionary to store results
        results = {}
        
        # Process each entry in the data
        for question_id, entry_info in tqdm(processing_info.items(), desc="Processing questions"):
            try:
                # Get image path and user prompt from processing info
                image_path = entry_info["image_path"]
                user_prompt = entry_info["user_prompt"]
                
                # Generate alternative questions
                raw_response = inference(
                    str(image_path), 
                    system_instruction, 
                    user_prompt
                )
                
                # Store the raw response for this question ID
                results[question_id] = raw_response
                
                # Periodically free GPU memory
                if torch.cuda.is_available() and (int(question_id) % 10 == 0):
                    torch.cuda.empty_cache()
                
                # After every 10 images, save partial results
                if int(question_id) % 10 == 0:
                    partial_output = format_output_json(data, results)
                    save_results_to_json(partial_output, output_json_path)
                    logger.info(f"Saved partial results up to question {question_id}")
                    
            except Exception as e:
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f"Error processing question ID {question_id}: {e}")
                results[question_id] = f"Error: {str(e)}"
                continue
        
        # Format the output according to the desired structure
        formatted_output = format_output_json(data, results)
        
        return formatted_output
        
    except Exception as e:
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"Error processing data: {e}")
        return {}

def main() -> None:
    """Main function to run the Qwen2.5-VL question generation pipeline."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Qwen2.5-VL Inference for Question Generation')
    
    # Add config file argument
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Required arguments (now not required in CLI, but required overall)
    parser.add_argument('--system_prompt', type=str, help='Path to system prompt file')
    parser.add_argument('--user_prompt', type=str, help='Path to user prompt template file')
    parser.add_argument('--image_folder', type=str, help='Path to folder containing images')
    parser.add_argument('--input_json', type=str, help='Path to input JSON file with questions')
    parser.add_argument('--output_json', type=str, help='Path to output JSON file for results')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to log file (default: logs/qwenvl.log)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Model name or path')
    parser.add_argument('--cache_dir', type=str, default='../../weight/vlm/qwen2.5-vl-3b-instruct',
                        help='Directory where model weights are cached')
    
    args = parser.parse_args()
    
    # Load YAML config if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            return
    
    # Convert args to a dictionary
    args_dict = vars(args)
    
    # Create a merged configuration with CLI arguments taking precedence
    merged_config = {}
    for key in args_dict:
        if key == 'config':  # Skip the config argument
            continue
        # Use CLI value if provided, otherwise use YAML value if present
        if args_dict[key] is not None:
            merged_config[key] = args_dict[key]
        elif key in config:
            merged_config[key] = config[key]
    
    # Check if all required arguments are present
    required_args = ['system_prompt', 'user_prompt', 'image_folder', 'input_json', 'output_json']
    missing_args = [arg for arg in required_args if arg not in merged_config]
    if missing_args:
        print(f"Missing required arguments: {', '.join(missing_args)}")
        parser.print_help()
        return
    
    # Setup logging with the specified path
    log_path = merged_config.get('log_path', 'logs/generation.log')
    setup_logging(log_path)
    
    # Set global device based on merged config
    global device
    device_str = merged_config.get('device', 'cuda')
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Set random seed
    seed = merged_config.get('seed', 42)
    set_seed(seed)
    
    try:
        # Load model and processor
        global model, processor
        model_name = merged_config.get('model_name', 'Qwen/Qwen2.5-VL-3B-Instruct')
        cache_dir = merged_config.get('cache_dir', '../../weight/vlm/qwen2.5-vl-3b-instruct')
        
        # Load model
        model, processor = load_model(model_name, cache_dir, device)
        
        # Read prompt files
        system_instruction = read_system_prompt(merged_config['system_prompt'])
        user_prompt_template = read_user_prompt(merged_config['user_prompt'])
        
        if not system_instruction:
            logger.warning("System prompt is empty. Using default instruction.")
            system_instruction = "You are a helpful assistant that generates alternative questions."
        
        if not user_prompt_template:
            logger.warning("User prompt template is empty. Using default template.")
            user_prompt_template = """
            {% for pair in qa_pairs %}
            Question ID: {{ pair.questionId }}
            Original Question: {{ pair.question }}
            {% endfor %}
            """
        
        # Process data and save partial results periodically
        results = read_and_process_data(
            merged_config['input_json'],
            merged_config['image_folder'],
            system_instruction,
            user_prompt_template,
            merged_config['output_json']
        )
        
        # Save final results
        save_results_to_json(results, merged_config['output_json'])
        
        logger.info(f"Processing completed successfully! Results saved to {merged_config['output_json']}")
    
    except Exception as e:
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
