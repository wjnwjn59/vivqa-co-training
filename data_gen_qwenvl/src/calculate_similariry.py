import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from pyvi import ViTokenizer
import argparse
from tqdm import tqdm
import concurrent.futures

def word_segment_vietnamese(text):
    """Segment Vietnamese text into words for PhoBERT processing"""
    try:
        return " ".join(word_tokenize(text))
    except ImportError:
        try:
            return ViTokenizer.tokenize(text)
        except ImportError:
            return text.replace(" ", "_")  # Basic fallback method

def compute_embeddings_batch(texts, tokenizer, model, device, max_position_embeddings, batch_size=32):
    """Compute embeddings for a batch of texts"""
    # Segment texts
    segmented_texts = [word_segment_vietnamese(text) for text in texts]
    
    # Process in batches
    all_embeddings = []
    
    for i in range(0, len(segmented_texts), batch_size):
        batch_texts = segmented_texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=max_position_embeddings
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)
    
    # Concatenate all batches
    return np.vstack(all_embeddings) if all_embeddings else np.array([])

def calculate_similarities_optimized(input_file, output_file):
    """Optimized version of similarity calculation"""
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dictionary to store results
    similarity_results = {}
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load PhoBERT model once
    print("Loading PhoBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)
    model.eval()
    max_position_embeddings = model.config.max_position_embeddings
    print("Model loaded successfully!")
    
    # Collect all unique questions for batch processing
    print("Collecting all questions...")
    all_questions = set()
    
    for entry in data.values():
        # Original questions
        for q_text in entry.get('original_question', {}).values():
            if q_text and q_text.strip():
                all_questions.add(q_text)
        
        # Paraphrased questions
        qwenvl_generated = entry.get('qwenvl_generated', {})
        if qwenvl_generated:
            for q_text in qwenvl_generated.get('question_paraphrased', {}).values():
                if q_text and q_text.strip():
                    all_questions.add(q_text)
            
            # Generated questions
            for q_text in qwenvl_generated.get('question_generated', {}).values():
                if q_text and q_text.strip():
                    all_questions.add(q_text)
    
    # Convert to list for batch processing
    print(f"Total unique questions: {len(all_questions)}")
    question_list = list(all_questions)
    
    # Compute all embeddings in batches
    print("Computing embeddings for all questions...")
    batch_size = 32  # Adjust based on GPU memory
    all_embeddings = compute_embeddings_batch(
        question_list, tokenizer, model, device, max_position_embeddings, batch_size
    )
    
    # Create a mapping from question to embedding
    question_to_embedding = {
        q: all_embeddings[i] for i, q in enumerate(question_list)
    }
    
    # Process each entry and calculate similarities
    print("Calculating similarities...")
    for entry_key in tqdm(data.keys(), desc="Processing entries", unit="entry"):
        entry = data[entry_key]
        entry_results = {
            "image_id": entry.get("image_id"),
            "generated_vs_original": {},
            "generated_vs_paraphrased": {}
        }
        
        # Get questions
        qwenvl_generated = entry.get('qwenvl_generated', {})
        if not qwenvl_generated:
            continue
            
        generated_questions = qwenvl_generated.get('question_generated', {})
        original_questions = entry.get('original_question', {})
        paraphrased_questions = qwenvl_generated.get('question_paraphrased', {})
        
        # Compare generated questions with original questions
        for gen_key, gen_question in generated_questions.items():
            if not gen_question or not gen_question.strip():
                continue
                
            entry_results["generated_vs_original"][gen_key] = {}
            
            gen_embedding = question_to_embedding[gen_question]
            
            for orig_key, orig_question in original_questions.items():
                if not orig_question or not orig_question.strip():
                    continue
                
                orig_embedding = question_to_embedding[orig_question]
                
                # Calculate similarity
                similarity = float(cosine_similarity(
                    gen_embedding.reshape(1, -1),
                    orig_embedding.reshape(1, -1)
                )[0][0])
                
                # Store result
                entry_results["generated_vs_original"][gen_key][orig_key] = {
                    "generated_question": gen_question,
                    "original_question": orig_question,
                    "similarity_score": similarity
                }
        
        # Compare generated questions with paraphrased questions
        for gen_key, gen_question in generated_questions.items():
            if not gen_question or not gen_question.strip():
                continue
                
            entry_results["generated_vs_paraphrased"][gen_key] = {}
            
            gen_embedding = question_to_embedding[gen_question]
            
            for para_key, para_question in paraphrased_questions.items():
                if not para_question or not para_question.strip():
                    continue
                
                para_embedding = question_to_embedding[para_question]
                
                # Calculate similarity
                similarity = float(cosine_similarity(
                    gen_embedding.reshape(1, -1),
                    para_embedding.reshape(1, -1)
                )[0][0])
                
                # Store result
                entry_results["generated_vs_paraphrased"][gen_key][para_key] = {
                    "generated_question": gen_question,
                    "paraphrased_question": para_question,
                    "similarity_score": similarity
                }
        
        similarity_results[entry_key] = entry_results
    
    # Save results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(similarity_results, f, ensure_ascii=False, indent=4)
    
    print(f"Calculated similarities for {len(similarity_results)} entries")
    return similarity_results

def main():
    parser = argparse.ArgumentParser(description='Calculate similarities between questions in JSON file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON file')

    args = parser.parse_args()

    similarity_results = calculate_similarities_optimized(args.input_file, args.output_file)
    print(f"Processed similarities for {len(similarity_results)} entries")

# Example usage with argparse
if __name__ == "__main__":
    main()
