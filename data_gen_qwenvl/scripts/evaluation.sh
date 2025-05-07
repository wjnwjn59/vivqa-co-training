# bash scripts/evaluation.sh
CUDA_VISIBLE_DEVICES=2 python model/qwenvl_evaluation.py --config config/evaluation_config.yaml
