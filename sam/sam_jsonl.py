import os
import cv2
import json
import numpy as np
from pycocotools import mask as mask_utils
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def generate_masks_to_jsonl(
    input_folder: str,
    output_jsonl_path: str,
    checkpoint_path: str,
    model_type: str = "vit_h",
    use_cuda: bool = False
):
    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    # Load the SAM model with the specified checkpoint and model type
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    if use_cuda:
        sam.to(device='cuda')  # Move model to GPU if requested
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Get list of image files in the input folder with common image extensions
    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    total_images = len(image_files)

    # Load already processed image IDs from the existing JSONL file, if any
    processed_files = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_files.add(data["image_id"])
                except:
                    continue  # Skip malformed lines

    # Open the output JSONL file in append mode to add new results
    with open(output_jsonl_path, 'a') as jsonl_file:
        for idx, filename in enumerate(image_files, start=1):
            # Skip files that have already been processed
            if filename in processed_files:
                print(f"Skipping (already processed): {filename}")
                continue

            print(f"Processing image {idx}/{total_images}: {filename}")

            # Read the image from disk
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot read image: {filename}")
                continue
            # Convert image from BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate masks for the image using SAM
            masks = mask_generator.generate(image)

            # Convert each mask to RLE (Run-Length Encoding) format for compact storage
            simplified_masks = []
            for mask in masks:
                binary_mask = mask['segmentation'].astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(binary_mask))
                rle['counts'] = rle['counts'].decode('utf-8')  # Decode bytes to string for JSON

                simplified_masks.append({
                    'bbox': mask['bbox'],           # Bounding box of the mask
                    'area': mask['area'],           # Area of the mask
                    'score': mask['predicted_iou'],# Confidence score
                    'rle': rle                     # Encoded mask
                })

            # Write the image_id and its masks to the output JSONL file
            json_line = {
                'image_id': filename,
                'masks': simplified_masks
            }
            jsonl_file.write(json.dumps(json_line) + '\n')
