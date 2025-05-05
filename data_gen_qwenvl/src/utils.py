# utils.py
from typing import Dict, Any, Optional, Tuple, List
import torch
import jinja2
import logging
import json
from pathlib import Path
import os
import re

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_system_prompt(file_path: str) -> str:
    """
    Reads the content of a system prompt file and returns it as a string.
    
    Args:
        file_path (str): Path to the system prompt text file
        
    Returns:
        str: Content of the file as a string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        return ""
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return ""

def read_user_prompt(file_path: str) -> str:
    """
    Reads the content of a user prompt template file and returns it as a string.
    
    Args:
        file_path (str): Path to the user prompt template file
        
    Returns:
        str: Content of the file as a string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        return ""
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return ""


# For paraphrase.py
def extract_questions_from_annotations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts question entries from a COCO-style annotation structure.

    Args:
        data (dict): JSON with 'annotations' and 'images'

    Returns:
        List[Dict]: Each dict contains:
            - questionId (str)
            - question (str)
            - image_id (int)
    """
    results = []
    annotations = data.get("annotations", {})

    for question_id, annotation in annotations.items():
        question = annotation.get("question", "").strip()
        image_index = annotation.get("image_id")

        if question:
            results.append({
                "questionId": question_id,
                "question": question,
                "image_id": image_index
            })

    return results

def extract_paraphrase(text: str):
    """
    Extracts the question ID and paraphrased question from the structured text response.
    Expected format:
    ID: [questionId]
    Paraphrased Question: [Your Paraphrased Vietnamese Question]
    
    Args:
        text (str): The text response from the model
        
    Returns:
        str or None: paraphrased_question if pattern matches, otherwise None
    """
    pattern = r"ID:\s*(\S+)\s*\nParaphrased Question:\s*(.+)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        paraphrased_question = match.group(2).strip()
        return paraphrased_question
    else:
        return None

def process_with_jinja_template(question_id: str, question: str, template_content: str) -> str:
    """
    Process a question using Jinja2 templating.
    
    Args:
        question_id (str): The ID of the question
        question (str): The question text
        template_content (str): The Jinja2 template content
        
    Returns:
        str: Rendered template as string
    """
    try:
        environment = jinja2.Environment()
        template = environment.from_string(template_content)
        questions = [{"questionId": question_id, "question": question}]
        user_prompt = template.render(questions=questions)
        return user_prompt
    except jinja2.exceptions.TemplateError as e:
        logger.error(f"Jinja2 template error: {e}")
        return f"Question ID: {question_id}\nOriginal Question: {question}"

def save_results_to_json(results: Dict[str, Any], output_file: str) -> None:
    """
    Save the processed results dictionary to a JSON file.
    
    Args:
        results (dict): The processed data with paraphrases
        output_file (str): The path to the output JSON file
    """
    try:
        output_path = Path(output_file)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Results successfully written to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")

def load_model(model_name: str, cache_dir: str, device) -> Tuple:
    """
    Load the Qwen2.5-VL model and processor.
    
    Args:
        model_name (str): Name or path of the model
        cache_dir (str): Directory to cache the model
        device: The device to load the model on
        
    Returns:
        tuple: (model, processor) loaded and configured
    """
    try:
        # Free up GPU memory
        torch.cuda.empty_cache()
        
        # Import here to avoid circular imports
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        # Load model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir
        ).to(device)
        
        # Set pixel range parameters for visual tokens
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=True,
            cache_dir=cache_dir,
        )
        
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def setup_logging(log_path: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_path (str): Path to log file
    """
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup file handler for logging
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add console handler for standard output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add handlers to logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    logger.info(f"Logging configured to {log_path}")

# For Generate.py
def extract_questions_from_nested_format(data: Dict[str, Any]) -> List[Dict]:
    """
    Extract questions from nested 'original_question' format to a flat list.
    """
    questions = []
    
    for question_id, entry in data.items():
        if "original_question" in entry:
            # Extract each question from the nested structure
            for q_key, q_text in entry["original_question"].items():
                if q_text.strip():  # Only include non-empty questions
                    # Create a unique sub-ID by combining the main ID and question number
                    sub_id = f"{question_id}_{q_key.replace('question_', '')}"
                    questions.append({
                        "questionId": sub_id,
                        "question": q_text,
                        "main_id": question_id  # Keep track of the parent ID
                    })
                    
    return questions

def process_nested_questions_with_template(data: Dict[str, Any], template_content: str, 
                                          image_folder: str) -> Dict[str, Dict]:
    """
    Process each entry in the nested question format and prepare for inference.
    """
    processing_info = {}
    environment = jinja2.Environment()
    template = environment.from_string(template_content)
    
    for question_id, entry in data.items():
        image_id = entry.get("image_id")
        image_path = os.path.join(image_folder, f"{int(image_id):012d}.jpg")
        
        # Extract questions for this entry
        entry_questions = []
        for q_key, q_text in entry["original_question"].items():
            if q_text.strip():  # Only include non-empty questions
                entry_questions.append({
                    "imageId": image_id,  # Changed from "questionId" to "imageId"
                    "question": q_text,
                    # Keep the original questionId for reference in processing_info
                    "_questionId": f"{question_id}_{q_key.replace('question_', '')}"
                })
        
        # Create user prompt using template - note the changed variable name (questions not qa_pairs)
        user_prompt = template.render(questions=entry_questions, image_source=image_path)
        
        # Store processing info
        processing_info[question_id] = {
            "image_id": image_id,
            "image_path": image_path,
            "questions": entry_questions,
            "user_prompt": user_prompt
        }
    
    return processing_info

def extract_generated_questions(text: str) -> list:
    """
    Extracts generated questions from the model response.
    
    Expected format:
    Generate Question 1: [Your 1st Generated Vietnamese Question]
    Generate Question 2: [Your 2nd Generated Vietnamese Question]
    Generate Question 3: [Your 3rd Generated Vietnamese Question]
    """
    pattern = r"Generate Question (\d+): (.+?)(?=Generate Question \d+:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return [question.strip() for _, question in matches]
    else:
        return []

def format_output_json(input_data: Dict[str, Any], 
                       generated_responses: Dict[str, str]) -> Dict[str, Any]:
    """
    Format the output JSON according to the specified structure.
    """
    output_data = {}
    for question_id, entry in input_data.items():
        output_entry = {
            "image_id": entry["image_id"],
            "original_question": entry["original_question"],
            "question_generated": {}
        }
        
        # Fix number of alternate questions to 3
        num_questions = 3
        
        # Get the model response for this entry
        response = generated_responses.get(question_id, "")
        
        # Extract the generated questions from the response
        pattern = r"Generate Question (\d+): (.+?)(?=Generate Question \d+:|$)"
        matches = re.findall(pattern, response, re.DOTALL)
        
        # Format extracted questions
        for i, (_, question) in enumerate(matches, 1):
            if i <= num_questions:
                alternate_key = f"alternate_question_{i}"
                output_entry["question_generated"][alternate_key] = question.strip()
        
        # Fill in any missing alternate questions up to 3
        for i in range(1, num_questions + 1):
            alternate_key = f"alternate_question_{i}"
            if alternate_key not in output_entry["question_generated"]:
                output_entry["question_generated"][alternate_key] = ""
        
        output_data[question_id] = output_entry
    
    return output_data


# For Evaluation.py 
def process_generated_questions_with_template(data: Dict[str, Any], template_content: str,
                                             image_folder: str) -> Dict[str, Dict]:
    """
    Process questions from the question_generated field and prepare for evaluation.
    
    Args:
        data: Input data with question_generated field
        template_content: Jinja2 template content
        image_folder: Path to the folder containing images
        
    Returns:
        Dict containing processed information for evaluation
    """
    processing_info = {}
    environment = jinja2.Environment()
    template = environment.from_string(template_content)

    for question_id, entry in data.items():
        image_id = entry.get("image_id")
        image_path = os.path.join(image_folder, f"{int(image_id):012d}.jpg")
        
        # Extract questions from question_generated instead of original_question
        entry_questions = []
        if "question_generated" in entry:
            for q_key, q_text in entry["question_generated"].items():
                if q_text.strip():  # Only include non-empty questions
                    entry_questions.append({
                        "imageId": image_id,
                        "question": q_text,
                        "_questionId": f"{question_id}_{q_key}"
                    })
            
            # Create user prompt using template
            user_prompt = template.render(questions=entry_questions, image_source=image_path)
            
            # Store processing info
            processing_info[question_id] = {
                "image_id": image_id,
                "image_path": image_path,
                "questions": entry_questions,
                "user_prompt": user_prompt
            }
        
    return processing_info
