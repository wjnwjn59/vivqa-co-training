# bash scripts/paraphrasing.sh
# cd /home/duyth/vqa_co_training/vivqa-co-training/data_gen_qwenvl
# export PYTHONPATH=$PWD:$PYTHONPATH
CUDA_VISIBLE_DEVICES=1 python model/qwenvl_paraphrasing.py --config config/paraphrasing_config.yaml