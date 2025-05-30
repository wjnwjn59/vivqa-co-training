#!/bin/bash
# Modify other parameters in configs directory
echo "Starting training with default arguments..."
python src/training/train.py \
  --dataset openvivqa \
  --root_dir /mnt/VLAI_data/OpenViVQA \
  --training_type base 
