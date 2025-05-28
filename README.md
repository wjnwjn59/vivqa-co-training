# ViVQA Co-training

## Installation

It is recommended to install all the dependencies and run this repo using conda environment (you can install conda following [here(https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)]):
```
conda create -n cotraining python=3.10 -y
conda activate cotraining

pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"
```

## Usage
To train a model with convetional supervised fine-tuning (SFT) setup on OpenViVQA dataset, run:
```
python src/training/train.py \
    --dataset openvivqa \
    --root-dir /path/to/dataset/dir \
    --training-type base
```

## To-do list
- [x] Baseline LoRA training for any VLM.
- [] Evaluation scripts for VQA.
- [] Baseline LoRA Co-training for any VLM.