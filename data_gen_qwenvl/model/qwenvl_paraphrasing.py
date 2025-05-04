from tqdm import tqdm
import json
import os
import torch
from typing import Dict, Any
from pathlib import Path
import argparse
import logging
import yaml
from qwen_vl_utils import process_vision_info
import sys

# Fix import path
from src.utils import (
    set_seed, read_system_prompt, read_user_prompt, extract_paraphrase,
    process_with_jinja_template, save_results_to_json, load_model, setup_logging
)

# Define module-level variables
model = None
processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logger = logging.getLogger(__name__)

def inference(system_instruction: str, user_prompt: str) -> str:
    """
    Perform inference with optimized performance.
    """
    global model, processor
    
    try:
        # Prepare messages for the system and the user
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
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
        
        # Use inference_mode instead of no_grad for better performance
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
        
        raw_response = output_text[0]
        
        # Extract the structured information
        paraphrased_question = extract_paraphrase(raw_response)
        return paraphrased_question if paraphrased_question else raw_response
        
    except Exception as e:
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"Error during inference: {e}")
        return f"Error: {str(e)}"

def read_and_process_data(json_path: str, system_instruction: str, user_prompt_template: str) -> Dict[str, Any]:
    """
    Read JSON data with question IDs as keys, process each entry, and
    generate paraphrased questions using the model.
    """
    try:
        # Try using faster JSON library if available
        try:
            import ujson as json_lib
        except ImportError:
            import json as json_lib
            
        # Convert paths to Path objects
        json_path = Path(json_path)
        
        # Read the JSON data
        with json_path.open("r", encoding="utf-8") as f:
            data = json_lib.load(f)
        
        # Create a dictionary to store results
        results: Dict[str, Any] = {}
        
        # Process each entry in the data
        for question_id, entry in tqdm(data.items(), desc="Processing questions"):
            try:
                # Extract fields
                image_id = entry.get("image_id")
                question = entry.get("question")
                answer = entry.get("answer")
                
                # Generate user prompt from template
                user_prompt = process_with_jinja_template(
                    question_id, question, user_prompt_template
                )
                
                # Get raw response from model
                raw_response = inference(system_instruction, user_prompt)
                
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
                
                # Periodically free GPU memory if using CUDA
                if torch.cuda.is_available() and (int(question_id) % 10 == 0):
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f"Error processing question ID {question_id}: {e}")
                continue
        
        return results
        
    except Exception as e:
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"Error processing data: {e}")
        return {}

def main() -> None:
    """Main function to run the Qwen2.5-VL paraphrasing pipeline."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Qwen2.5-VL Inference for Question Paraphrasing')
    
    # New: Add config flag
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Required arguments (no longer marked as required since they might be in the config file)
    parser.add_argument('--system_prompt', type=str,
                        help='Path to system prompt file')
    parser.add_argument('--user_prompt', type=str,
                        help='Path to user prompt template file')
    parser.add_argument('--input_json', type=str,
                        help='Path to input JSON file with questions')
    parser.add_argument('--output_json', type=str,
                        help='Path to output JSON file for results')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_path', type=str, default='logs/qwenvl.log',
                        help='Path to log file (default: logs/qwenvl.log)')
    args = parser.parse_args()
    
    # Load YAML config if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            # Override args with YAML values
            for key, value in config.items():
                setattr(args, key, value)
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            return
    
    # Check if required arguments are provided
    required_args = ['system_prompt', 'user_prompt', 'input_json', 'output_json']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        print(f"Missing required arguments: {', '.join(missing_args)}")
        parser.print_help()
        return
    
    # Setup logging with the specified path
    setup_logging(args.log_path)
    
    # Set global device based on args
    global device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for inference")
    
    # Set random seed
    set_seed(args.seed)
    
    try:
        # Load model and processor
        global model, processor
        model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
        cache_dir = '../../weight/vlm/qwen2.5-vl-3b-instruct'
        
        # Fix: Add missing device parameter
        model, processor = load_model(model_name, cache_dir, device)
        
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
            system_instruction,
            user_prompt_template
        )
        
        # Save results
        save_results_to_json(results, args.output_json)
        
        logger.info(f"Processing completed successfully. Results saved to {args.output_json}")
        
    except Exception as e:
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"Error in main execution: {e}")
        raise
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
