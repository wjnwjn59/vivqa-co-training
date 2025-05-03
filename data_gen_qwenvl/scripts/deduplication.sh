# bash scripts/deduplication.sh
python src/phobert_duplication.py --input_json data/qwenvl_openvivqa/qwenvl_train.json --output_json data/duplicated/duplicated_train.json --threshold 0.9