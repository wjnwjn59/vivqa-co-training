# bash scripts/paraphrasing.sh
# export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=1 python model/qwenvl_paraphrasing.py --config config/paraphrasing_config.yaml