import os
import cv2
import json
import numpy as np
import random
from pycocotools import mask as mask_utils

def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply a single binary mask on the image with a given RGB color and transparency (alpha).
    """
    for c in range(3):  # For each channel (R, G, B)
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c],
            image[:, :, c]
        )
    return np.clip(image, 0, 255).astype(np.uint8)

def visualize_masks_from_jsonl(
    jsonl_path: str,
    input_folder: str,
    output_folder: str,
    alpha: float = 0.3,
    max_images: int = None
):
    """
    Read RLE masks from a JSONL file, decode them, overlay them on corresponding images, and save the visualized images.

    Args:
        jsonl_path (str): Path to the JSONL file containing image_id and RLE masks.
        input_folder (str): Folder containing the original images.
        output_folder (str): Folder where the visualized images will be saved.
        alpha (float): Transparency level for the overlaid masks.
        max_images (int): Optional. Maximum number of images to process.
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if max_images is not None and i >= max_images:
                break

            data = json.loads(line)
            filename = data['image_id']
            masks = data['masks']

            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot read image: {filename}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for m in masks:
                rle = m['rle']
                mask = mask_utils.decode(rle)

                # Resize mask if its size doesn't match the image
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                color = [random.randint(0, 255) for _ in range(3)]
                image = apply_mask(image, mask, color, alpha=alpha)

            save_path = os.path.join(output_folder, f"masked_{filename}")
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)