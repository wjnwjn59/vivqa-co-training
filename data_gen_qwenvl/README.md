# QwenVL 2.5 3B Instruct OpenViVQA Data Generating Project

This project provides a pipeline for processing, generating, evaluating, and managing Vietnamese visual question-answering (VQA) data using the Qwen2.5-VL multimodal model and PhoBERT for semantic similarity detection.

## Project Structure

```
project/
├── data/                   # Data storage directory
├── logs/                   # Log files
│   ├── evaluation.log
│   ├── generation.log
│   └── paraphrasing.log
├── model/                  # Model files
│   ├── __init__.py
│   ├── qwenvl_evaluation.py
│   ├── qwenvl_generation.py
│   └── qwenvl_paraphrasing.py
├── scripts/                # Shell scripts for running pipelines
│   ├── deduplication.sh
│   ├── evaluation.sh
│   ├── extract_q.sh
│   ├── generation.sh
│   └── paraphrasing.sh
├── src/                    # Source code
│   ├── phobert_duplication.py
│   ├── q_extraction.py
│   └── utils.py
├── templates/              # Prompt templates
│   ├── evaluate_q/
│   ├── generate_q/
│   └── paraphase_q/
├── .gitignore
├── README.md
└── requirements.txt
```

## Components

### 1. Question Extraction

The `q_extraction.py` script extracts questions from existing datasets and organizes them by image ID, creating a structured format for further processing.

```bash
python src/q_extraction.py --input_json INPUT_JSON --output_json OUTPUT_JSON
```

### 2. Question Generation

`qwenvl_generation.py` generates alternative questions for existing visual questions using the Qwen2.5-VL model.

```bash
python model/qwenvl_generation.py --config generation.yaml
```

### 3. Question Evaluation

`qwenvl_evaluation.py` evaluates generated questions based on linguistic quality and image grounding, producing two output files: qualified and non-qualified questions.

```bash
python model/qwenvl_evaluation.py --config evaluation.yaml
```

### 4. Question Paraphrasing

`qwenvl_paraphrasing.py` creates paraphrased versions of questions while maintaining their meaning.

```bash
python model/qwenvl_paraphrasing.py --config paraphrasing.yaml
```

### 5. Duplicate Detection

`phobert_duplication.py` uses PhoBERT embeddings to identify semantically similar questions.

```bash
python src/phobert_duplication.py --input_file INPUT_FILE --output_file OUTPUT_FILE --threshold 0.9
```

## Configuration

Example YAML configuration for evaluation:

```yaml
# evaluation.yaml
system_prompt: templates/evaluate_q/system_prompt.txt
user_prompt: templates/evaluate_q/user_prompt.txt
image_folder: /path/to/images/
input_json: data/generated/generated_train.json
output_dir: data/evaluated/

device: cuda
seed: 42
log_path: logs/evaluation.log
model_name: Qwen/Qwen2.5-VL-3B-Instruct
cache_dir: ../../weight/vlm/qwen2.5-vl-3b-instruct
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```
## Usage

1. Extract questions from your dataset
2. Generate alternative questions
3. Evaluate question quality
4. Identify and remove duplicates

You can use the shell scripts in the `scripts/` directory to run these steps in sequence.
