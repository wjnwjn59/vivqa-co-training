# utils.py
from typing import Dict, Any, Optional, Tuple
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
        qa_pairs = [{"questionId": question_id, "question": question}]
        user_prompt = template.render(qa_pairs=qa_pairs)
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
