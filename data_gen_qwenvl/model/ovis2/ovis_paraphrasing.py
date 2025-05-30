from tqdm import tqdm
import json
import os
import torch
from typing import Dict, Any
from pathlib import Path
import argparse
import logging
import yaml
import sys

# Fix import path
from src.utils import (
    set_seed, read_system_prompt, read_user_prompt, extract_paraphrase, extract_questions_from_annotations,
    process_with_jinja_template, save_results_to_json, load_model_ovis, setup_logging, load_existing_results,
    filter_unprocessed_questions
)

# Define module-level variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logger = logging.getLogger(__name__)

def inference(system_instruction: str, user_prompt: str) -> str:
    global model
    try:

        # Ovis2 format: prompts
        query = f"{system_instruction}\n{user_prompt}"

        # Tokenize using Ovis2 API
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        max_partition = 9
        logger.error("DEBUG:", model.preprocess_inputs(query, images=None, max_partition=max_partition))

        _, input_ids, pixel_values = model.preprocess_inputs(
            query, images=None, max_partition=max_partition
        )

        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
            pixel_values = [pixel_values]
        else:
            pixel_values = None
            
        # Inference
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=128,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs
            )[0]

        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        paraphrased_question = extract_paraphrase(output)
        return paraphrased_question if paraphrased_question else output

    except Exception as e:
        logger.error(f"Error during Ovis2 inference: {e}")
        return f"Error: {str(e)}"

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
    parser = argparse.ArgumentParser(description='Ovis2 Inference for Question Paraphrasing')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--system_prompt', type=str, help='Path to system prompt file')
    parser.add_argument('--user_prompt', type=str, help='Path to user prompt template file')
    parser.add_argument('--input_json', type=str, help='Path to input JSON file with questions')
    parser.add_argument('--output_json', type=str, help='Path to output JSON file for results')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run inference on (default: cuda)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file (default: logs/qwenvl.log)')
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

    setup_logging(args.log_path or 'logs/ovis2.log')
    global device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for inference")

    set_seed(args.seed or 42)

    try:
        global model
        model_name = args.model_name or 'AIDC-AI/Ovis2-8B'
        cache_dir = args.cache_dir or '../../weight/vlm/ovis2-8b'
        model = load_model_ovis(model_name, cache_dir, device)

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
