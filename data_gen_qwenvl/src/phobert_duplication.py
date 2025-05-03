import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from pyvi import ViTokenizer
import argparse

def word_segment_vietnamese(text):
    """Segment Vietnamese text into words for PhoBERT processing"""
    try:
        return " ".join(word_tokenize(text))
    except ImportError:
        try:
            return ViTokenizer.tokenize(text)
        except ImportError:
            return text.replace(" ", "_")  # Basic fallback method

def compute_phobert_embeddings(texts):
    """Compute embeddings using PhoBERT model"""
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModel.from_pretrained("vinai/phobert-base-v2")
    model.eval()
    
    max_position_embeddings = model.config.max_position_embeddings
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            segmented_text = word_segment_vietnamese(text)
            
            inputs = tokenizer(
                segmented_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=max_position_embeddings
            )
            
            outputs = model(**inputs)
            # Use mean pooling of last hidden states as embedding
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    return np.array(embeddings)

def calculate_similarity(question1, question2):
    """Calculate semantic similarity between two questions using PhoBERT"""
    # Compute embeddings for both questions
    embeddings = compute_phobert_embeddings([question1, question2])
    
    # Calculate cosine similarity between the embeddings
    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]
    
    return similarity

def find_duplicated_questions(input_file, output_file, threshold=0.9):
    """Find duplicated questions in the input JSON file and save results to output"""
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dictionary to store results
    duplicated_data = {}
    
    # Process each entry
    for key, entry in data.items():
        # Get questions from question_generated field
        questions = entry.get('question_generated', {})
        
        # Filter out empty questions
        filtered_questions = {k: v for k, v in questions.items() if v and v.strip()}
        
        # Get question values
        q_values = list(filtered_questions.values())
        
        # Skip if there are less than 2 questions
        if len(q_values) < 2:
            continue
        question_items = list(filtered_questions.items())
        # Compare each pair of questions
        found_duplicate = False
        for i in range(len(question_items)):
            if found_duplicate:
                break
                
            for j in range(i+1, len(question_items)):
                q1_key, q1_value = question_items[i]
                q2_key, q2_value = question_items[j]
                
                # Skip if either key is "question_paraphrase"
                if q1_key == "question_paraphrase" or q2_key == "question_paraphrase":
                    continue
                
                # Check for exact match first
                if q1_value == q2_value:
                    duplicated_data[key] = {
                        "image_id": entry.get("image_id"),
                        "duplicated_question": {
                            q1_key: q1_value,
                            q2_key: q2_value
                        }
                    }
                    found_duplicate = True
                    break
                
                # Calculate similarity for non-identical questions
                similarity = calculate_similarity(q1_value, q2_value)
                
                # If similarity is above threshold, add to duplicates
                if similarity > threshold:
                    duplicated_data[key] = {
                        "image_id": entry.get("image_id"),
                        "duplicated_question": {
                            q1_key: q1_value,
                            q2_key: q2_value
                        }
                    }
                    found_duplicate = True
                    break
    
    # Save results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(duplicated_data, f, ensure_ascii=False, indent=4)
    
    print(f"Found duplicates in {len(duplicated_data)} entries")
    return duplicated_data

def main():
    parser = argparse.ArgumentParser(description='Find duplicated questions in JSON file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--threshold', type=float, default=0.9, help='Similarity threshold for duplicates')

    args = parser.parse_args()

    duplicated = find_duplicated_questions(args.input_file, args.output_file, args.threshold)
    print(f"Found duplicates in {len(duplicated)} entries")

# Example usage with argparse
if __name__ == "__main__":
    main()