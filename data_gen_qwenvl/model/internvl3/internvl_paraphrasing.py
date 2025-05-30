from typing import Dict, Any, List, Tuple
from pathlib import Path
import argparse
import logging
import yaml
from tqdm import tqdm
import json
import os
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix import path
from src.utils import (
    set_seed, read_system_prompt, read_user_prompt, extract_paraphrase, extract_questions_from_annotations,
    process_with_jinja_template, save_results_to_json, load_model_internvl, setup_logging, load_existing_results,
    filter_unprocessed_questions
)

# Define module-level variables
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def inference(system_instruction: str, user_prompt: str) -> str:
    """
    InternVL-3 single-image VQA inference.
    """
    
    prompt = (
        f"{system_instruction}\n"
        f"Question: {user_prompt}\n"
    )
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    model.eval()
    with torch.no_grad():
        response = model.chat(
            tokenizer,
            None,
            prompt,
            generation_config
        )

    raw_response = response.split("Answer:")[-1].strip()
    paraphrased_question = extract_paraphrase(raw_response)
    return paraphrased_question if paraphrased_question else raw_response

def read_and_process_data(json_path: str, system_instruction: str, user_prompt_template: str, output_json_path: str) -> Dict[str, Any]:
    try:
        import json as json_lib
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as f:
            data = json_lib.load(f)
        questions_data = extract_questions_from_annotations(data)

        # Load previous results and filter unprocessed
        results = load_existing_results(output_json_path)
        questions_data = filter_unprocessed_questions(questions_data, results)

        for idx, item in enumerate(tqdm(questions_data, desc="Processing questions")):
            question_id = str(item["questionId"])
            question = item["question"]
            image_id = item.get("image_id")
            user_prompt = process_with_jinja_template(question_id, question, user_prompt_template)
            raw_response = inference(system_instruction, user_prompt)
            paraphrased_question = extract_paraphrase(raw_response)
            paraphrase = paraphrased_question if paraphrased_question else raw_response
            results[question_id] = {
                "image_id": image_id,
                "question": question,
                "question_generated": {
                    "question_paraphrased": paraphrase
                }
            }

            if torch.cuda.is_available() and (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
            if (idx + 1) % 20 == 0:
                save_results_to_json(results, output_json_path)
                logger.info(f"Autosaved {idx + 1} questions to {output_json_path}")

        return results
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description='Qwen2.5-VL Inference for Question Paraphrasing')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--system_prompt', type=str, help='Path to system prompt file')
    parser.add_argument('--user_prompt', type=str, help='Path to user prompt template file')
    parser.add_argument('--input_json', type=str, help='Path to input JSON file with questions')
    parser.add_argument('--output_json', type=str, help='Path to output JSON file for results')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run inference on')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file')
    parser.add_argument('--model_name', type=str, default=None, help='Model name or path')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory where model weights are cached')
    args = parser.parse_args()

    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            return

    required_args = ['system_prompt', 'user_prompt', 'input_json', 'output_json']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        print(f"Missing required arguments: {', '.join(missing_args)}")
        parser.print_help()
        return

    setup_logging(args.log_path or 'logs/internvl.log')
    global device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for inference")

    set_seed(args.seed or 42)

    try:
        global model, tokenizer
        model_name = args.model_name or "OpenGVLab/InternVL3-8B"
        cache_dir = args.cache_dir or "../../weight/vlm/internvl3-8b"
        model, tokenizer = load_model_internvl(model_name, cache_dir, device)

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

        results = read_and_process_data(
            args.input_json,
            system_instruction,
            user_prompt_template,
            args.output_json
        )
        save_results_to_json(results, args.output_json)
        logger.info(f"Processing completed successfully. Results saved to {args.output_json}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
