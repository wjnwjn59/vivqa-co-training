# bash scripts/generation.sh
# cd /home/duyth/vqa_co_training/vivqa-co-training/data_gen_qwenvl
# export PYTHONPATH=$PWD:$PYTHONPATH
CUDA_VISIBLE_DEVICES=1 python model/qwenvl_generation.py --config config/generation_config.yaml