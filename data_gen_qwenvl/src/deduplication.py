import json

def find_high_similarity_questions(json_data, threshold=0.9):
    result = {}
    for image_id, image_data in json_data.items():
        for category in ['generated_vs_original', 'generated_vs_paraphrased']:
            if category in image_data:
                for gen_q_key, questions in image_data[category].items():
                    for q_key, details in questions.items():
                        if details['similarity_score'] > threshold:
                            if image_id not in result:
                                result[image_id] = []
                            if gen_q_key not in result[image_id]:
                                result[image_id].append(gen_q_key)
    return result

def remove_duplicates(origin_json, duplicated_json):
    filtered_json = {}
    for image_id, content in origin_json.items():
        filtered_content = json.loads(json.dumps(content))
        if "qwenvl_generated" in filtered_content and "question_generated" in filtered_content["qwenvl_generated"]:
            generated_questions = filtered_content["qwenvl_generated"]["question_generated"]
            if image_id in duplicated_json:
                for gen_q_key in duplicated_json[image_id]:
                    if gen_q_key in generated_questions:
                        del generated_questions[gen_q_key]
        filtered_json[image_id] = filtered_content
    return filtered_json

def main():
    with open('data/duplicated/score_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    high_similarity = find_high_similarity_questions(data)
    
    # Write duplicated json and ensure file is closed before reading
    with open('data/duplicated/duplicated_train.json', 'w', encoding='utf-8') as f:
        json.dump(high_similarity, f, indent=4, ensure_ascii=False)
    
    # Now read the duplicated json after the write is complete
    with open('data/duplicated/duplicated_train.json', 'r', encoding='utf-8') as f:
        duplicated_json = json.load(f)
        
    with open('data/qwenvl_openvivqa/qwenvl_train.json', 'r', encoding='utf-8') as f:
        origin_json = json.load(f)
        
    filtered_json = remove_duplicates(origin_json, duplicated_json)
    
    with open('data/qwenvl_openvivqa/qwenvl_train_filtered.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_json, f, indent=4, ensure_ascii=False)
        
    return high_similarity, filtered_json

if __name__ == "__main__":
    main()
