import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from pyvi import ViTokenizer

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

# Example usage
if __name__ == "__main__":
    question1 = "phía dưới dòng chữ 24/24 có hình gì ?"
    question2 = "Hình ảnh gì nằm bên dưới dòng chữ 24/24?"
    
    similarity_score = calculate_similarity(question1, question2)
    print(f"Similarity between the two questions: {similarity_score:.4f}")
