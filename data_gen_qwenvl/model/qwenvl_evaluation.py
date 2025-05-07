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
import json as json_lib

from src.utils import (
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


def extract_evaluation_results(response_text: str) -> List[Dict[str, Any]]:
    """Extract evaluation results for multiple questions from model response."""
    # Split the response by question blocks
    question_blocks = re.split(r'Generated Question:', response_text)[1:]  # Skip the first empty part
    
    results = []
    image_id = None
    
    # Extract image ID if present
    id_match = re.search(r"ID:\s*(\d+)", response_text)
    if id_match:
        image_id = int(id_match.group(1))
    
    # Process each question block
    for block in question_blocks:
        result = {}
        if image_id:
            result["ID"] = image_id
        
        # Extract question text
        question_text = block.split('\n')[0].strip()
        result["question_text"] = question_text
        
        # Extract reason
        reason_match = re.search(r"Reason:\s*(.*?)(?=Linguistic Score:|$)", block, re.DOTALL)
        if reason_match:
            result["reason"] = reason_match.group(1).strip()
        
        # Extract scores
        linguistic_match = re.search(r"Linguistic Score:\s*(\d+(?:\.\d+)?)", block)
        if linguistic_match:
            result["linguistic_score"] = float(linguistic_match.group(1))
        
        grounding_match = re.search(r"Image Grounding Score:\s*(\d+(?:\.\d+)?)", block)
        if grounding_match:
            result["image_grounding_score"] = float(grounding_match.group(1))
        
        # Calculate final score
        if "linguistic_score" in result and "image_grounding_score" in result:
            result["final_score"] = 0.2 * result["linguistic_score"] + 0.8 * result["image_grounding_score"]
        
        results.append(result)
    
    return results


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
                         user_prompt_template: str, output_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process questions grouped by image_id and evaluate them."""
    try:
        json_path = Path(json_path)
        image_folder = Path(image_folder)
        processed_images = 0
        with json_path.open("r", encoding="utf-8") as f:
            data = json_lib.load(f)
        
        # Ensure output directory exists
        output_folder     = Path(output_path)
        output_folder.mkdir(parents=True, exist_ok=True)
        qualified_path     = output_folder / "qualified_questions_train.json"
        not_qualified_path = output_folder / "not_qualified_questions_train.json"
        
        # Use the function to process generated questions - grouped by image_id
        processing_info = process_generated_questions_with_template(data, user_prompt_template, str(image_folder))
        qualified_questions = []
        not_qualified_questions = []
        
        # Process by image_id
        for image_id, entry_info in tqdm(processing_info.items(), desc="Evaluating images"):
            try:
                image_path = entry_info["image_path"]
                user_prompt = entry_info["user_prompt"]
                questions_list = entry_info["questions"]
                
                # Do inference with all questions for this image
                raw_response = inference(
                    str(image_path),
                    system_instruction,
                    user_prompt
                )

                # Extract evaluation results for all questions
                question_evaluations = extract_evaluation_results(raw_response)

                # Match extracted evaluations to questions based on text similarity
                for question_data in questions_list:
                    question = question_data["question"]
                    # Use get() to safely access the key or fall back to another field
                    question_key = question_data.get("key", question_data.get("questionId", "generated_question"))
                    
                    # Find matching evaluation
                    matching_eval = None
                    for eval_result in question_evaluations:
                        if question.lower() in eval_result.get("question_text", "").lower():
                            matching_eval = eval_result
                            break
                    
                    if not matching_eval:
                        # If no match found, use empty values
                        matching_eval = {"reason": "", "linguistic_score": 0, "image_grounding_score": 0, "final_score": 0}
                    
                    # Create result for this question
                    result = {
                        "ID": int(image_id),
                        question_key: question,
                        "reason": matching_eval.get("reason", ""),
                        "linguistic_score": int(matching_eval.get("linguistic_score", 0)),
                        "image_grounding_score": int(matching_eval.get("image_grounding_score", 0)),
                        "final_score": int(matching_eval.get("final_score", 0))
                    }
                    
                    # Categorize based on final score
                    if result["final_score"] >= 7:
                        qualified_questions.append(result)
                    else:
                        not_qualified_questions.append(result)

                processed_images += 1
                
                # Save after every 10 images
                if processed_images % 10 == 0:
                    with qualified_path.open("w", encoding="utf-8") as f:
                        json.dump(qualified_questions, f, ensure_ascii=False, indent=2)
                    
                    with not_qualified_path.open("w", encoding="utf-8") as f:
                        json.dump(not_qualified_questions, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Saved results after processing {processed_images} images")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing image ID {image_id}: {e}")
                
                # Create error entries for each question in this image
                if "questions" in entry_info:
                    for question_data in entry_info["questions"]:
                        question = question_data["question"]
                        # Use same approach for error handling
                        question_key = question_data.get("key", question_data.get("questionId", "generated_question"))
                        
                        not_qualified_questions.append({
                            "ID": int(image_id),
                            question_key: question,
                            "reason": f"Error: {str(e)}",
                            "linguistic_score": 0,
                            "image_grounding_score": 0,
                            "final_score": 0
                        })
        
        return qualified_questions, not_qualified_questions
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return [], []


    
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
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    
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
            user_prompt_template,
            merged_config['output_dir'] 
        )
        
        # Save results

        
        logger.info(f"Evaluation completed! Qualified: {len(qualified_questions)}, Not qualified: {len(not_qualified_questions)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
