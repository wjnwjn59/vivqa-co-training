from tqdm import tqdm
import json
import os
import re
import torch
from typing import Dict, Any, List, Tuple
from pathlib import Path
import argparse
import logging
import yaml
from qwen_vl_utils import process_vision_info

from utils import (
    set_seed, read_system_prompt, read_user_prompt, 
    load_model, setup_logging, process_generated_questions_with_template
)

# Define global variables
model = None
processor = None
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(image_path: str, system_instruction: str, user_prompt: str) -> str:
    """Perform inference with the model."""
    global model, processor
    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": user_prompt},
            ]},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,  # Increased for evaluation output
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return f"Error during inference: {str(e)}"


def extract_evaluation_result(response_text: str) -> Dict[str, Any]:
    """Extract evaluation results from model response."""
    result = {}
    
    id_match = re.search(r"ID:\s*(\S+)", response_text)
    if id_match:
        result["id"] = id_match.group(1).strip()
    
    reason_match = re.search(r"Reason:\s*(.*?)(?=Linguistic Score:|$)", response_text, re.DOTALL)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()
    
    linguistic_match = re.search(r"Linguistic Score:\s*(\d+(?:\.\d+)?)", response_text)
    if linguistic_match:
        result["linguistic_score"] = float(linguistic_match.group(1))
    
    grounding_match = re.search(r"Image Grounding Score:\s*(\d+(?:\.\d+)?)", response_text)
    if grounding_match:
        result["image_grounding_score"] = float(grounding_match.group(1))
    
    # Calculate final score
    if "linguistic_score" in result and "image_grounding_score" in result:
        result["final_score"] = 0.2 * result["linguistic_score"] + 0.8 * result["image_grounding_score"]
    
    return result

def save_results_to_json(data: Dict[str, Any], output_file: str) -> None:
    """Save results to JSON file."""
    try:
        output_path = Path(output_file)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")

def read_and_process_data(json_path: str, image_folder: str, system_instruction: str, 
                         user_prompt_template: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process questions and evaluate them."""
    try:
        try:
            import ujson as json_lib
        except ImportError:
            import json as json_lib
        
        json_path = Path(json_path)
        image_folder = Path(image_folder)
        
        with json_path.open("r", encoding="utf-8") as f:
            data = json_lib.load(f)
        
        # Use the new function to process generated questions instead
        processing_info = process_generated_questions_with_template(data, user_prompt_template, str(image_folder))
        
        qualified_questions = {}
        not_qualified_questions = {}
        
        for question_id, entry_info in tqdm(processing_info.items(), desc="Evaluating questions"):
            try:
                image_path = entry_info["image_path"]
                user_prompt = entry_info["user_prompt"]
                
                raw_response = inference(
                    str(image_path),
                    system_instruction,
                    user_prompt
                )
                
                evaluation_result = extract_evaluation_result(raw_response)
                
                if "final_score" in evaluation_result:
                    # Include both original and generated questions in the result
                    evaluation_result["original_question"] = data[question_id]["original_question"]
                    evaluation_result["question_generated"] = data[question_id]["question_generated"]
                    evaluation_result["image_id"] = data[question_id]["image_id"]
                    
                    # Sort based on final score
                    if evaluation_result["final_score"] > 8:
                        qualified_questions[question_id] = evaluation_result
                    else:
                        not_qualified_questions[question_id] = evaluation_result
                else:
                    logger.warning(f"Failed to extract evaluation for question ID {question_id}")
                    not_qualified_questions[question_id] = {
                        "original_question": data[question_id]["original_question"],
                        "question_generated": data[question_id]["question_generated"],
                        "image_id": data[question_id]["image_id"],
                        "error": "Failed to extract evaluation"
                    }
                
                if torch.cuda.is_available() and (int(question_id) % 10 == 0):
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing question ID {question_id}: {e}")
                not_qualified_questions[question_id] = {
                    "original_question": data[question_id]["original_question"],
                    "question_generated": data[question_id]["question_generated"],
                    "image_id": data[question_id]["image_id"],
                    "error": str(e)
                }
        
        return qualified_questions, not_qualified_questions
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return {}, {}
    
def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description='Qwen2.5-VL Question Evaluation')
    
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--system_prompt', type=str, help='Path to system prompt file')
    parser.add_argument('--user_prompt', type=str, help='Path to user prompt template file')
    parser.add_argument('--image_folder', type=str, help='Path to folder containing images')
    parser.add_argument('--input_json', type=str, help='Path to input JSON file with questions')
    parser.add_argument('--output_dir', type=str, help='Directory to save output JSON files')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_path', type=str, default='logs/evaluation.log')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='../../weight/vlm/qwen2.5-vl-3b-instruct')
    
    args = parser.parse_args()
    
    # Process config and arguments
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    args_dict = vars(args)
    merged_config = {}
    for key in args_dict:
        if key == 'config': continue
        if args_dict[key] is not None:
            merged_config[key] = args_dict[key]
        elif key in config:
            merged_config[key] = config[key]
    
    # Validate required arguments
    required_args = ['system_prompt', 'user_prompt', 'image_folder', 'input_json', 'output_dir']
    missing_args = [arg for arg in required_args if arg not in merged_config]
    if missing_args:
        print(f"Missing required arguments: {', '.join(missing_args)}")
        parser.print_help()
        return
    
    # Setup environment
    setup_logging(merged_config.get('log_path', 'logs/evaluation.log'))
    global device
    device_str = merged_config.get('device', 'cuda')
    device = torch.device('cuda' if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
    set_seed(merged_config.get('seed', 42))
    
    try:
        # Load model and processor
        global model, processor
        model_name = merged_config.get('model_name', 'Qwen/Qwen2.5-VL-3B-Instruct')
        cache_dir = merged_config.get('cache_dir', '../../weight/vlm/qwen2.5-vl-3b-instruct')
        model, processor = load_model(model_name, cache_dir, device)
        
        # Read prompts
        system_instruction = read_system_prompt(merged_config['system_prompt'])
        user_prompt_template = read_user_prompt(merged_config['user_prompt'])
        
        # Process data
        qualified_questions, not_qualified_questions = read_and_process_data(
            merged_config['input_json'],
            merged_config['image_folder'],
            system_instruction,
            user_prompt_template
        )
        
        # Save results
        save_results_to_json(qualified_questions, "data/evaluated/qualified_q.json")
        save_results_to_json(not_qualified_questions, "data/evaluated/not_qualified_q.json")
        
        logger.info(f"Evaluation completed! Qualified: {len(qualified_questions)}, Not qualified: {len(not_qualified_questions)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
