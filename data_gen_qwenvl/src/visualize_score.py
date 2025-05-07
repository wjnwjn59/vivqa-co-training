import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import defaultdict

def parse_json_file(file_path):
    """Parse JSON file with potential text before the JSON data"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find start of JSON object
    start_index = content.find('{')
    if start_index == -1:
        raise ValueError("No JSON data found")
    
    json_str = content[start_index:]
    return json.loads(json_str)

def visualize_similarity_scores(json_file, output_file='data/duplicated/similarity_distribution.png'):
    """Create visualization of all similarity scores in the JSON data"""
    # Parse JSON data
    data = parse_json_file(json_file)
    
    # Extract all similarity scores
    original_scores = []
    paraphrased_scores = []
    
    # Process each image_id
    for image_id, image_data in data.items():
        # Extract original question scores
        if "generated_vs_original" in image_data:
            for gen_q_key, comparisons in image_data["generated_vs_original"].items():
                for q_key, details in comparisons.items():
                    original_scores.append(details["similarity_score"])
        
        # Extract paraphrased question scores
        if "generated_vs_paraphrased" in image_data:
            for gen_q_key, comparisons in image_data["generated_vs_paraphrased"].items():
                for q_key, details in comparisons.items():
                    paraphrased_scores.append(details["similarity_score"])
    
    # Create the histogram visualization
    plt.figure(figsize=(12, 6))
    
    bins = np.linspace(0.45, 1.0, 25)
    plt.hist(original_scores, bins=bins, alpha=0.7, label='Original Questions', color='skyblue')
    plt.hist(paraphrased_scores, bins=bins, alpha=0.7, label='Paraphrased Questions', color='lightgreen')
    
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Question Similarity Scores', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    
    print(f"Visualization saved as {output_file}")
    return output_file

# Example usage
visualize_similarity_scores("data/duplicated/score_dev.json")
