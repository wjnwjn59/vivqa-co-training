# Segment Anything Mask Pipeline

This project provides a simple Python pipeline to:

1. Automatically **generate segmentation masks** using Meta's Segment Anything Model (SAM).
2. **Visualize masks** on images using random colors and alpha blending.
3. Save results in compact COCO RLE format (`.jsonl` file).

**The code requires python>=3.8, as well as pytorch>=1.7 and torchvision>=0.8.**
---

## Dependencies

- `opencv-python`
- `numpy`
- `pycocotools`
- `segment-anything`

Install Segment Anything:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install dependencies:

```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Download Model Checkpoints

To use the Segment Anything Model (SAM), you need to download one of the official pre-trained checkpoints from Meta:

**Available at**:  
[https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)

**Recommended: ViT-H Checkpoint**
---

## Usage

### 1. Generate RLE Masks

```python
from generate_masks import generate_masks_to_jsonl

generate_masks_to_jsonl(
    input_folder="your_image_folder",
    output_jsonl_path="output_masks.jsonl",
    checkpoint_path="path_to_sam_checkpoint.pth",
    model_type="vit_h",
    use_cuda=True
)
```

This creates a `.jsonl` file with COCO-style RLE masks for each image.

---

### 2. Visualize Masks

```python
from visualize_masks import visualize_masks_from_jsonl

visualize_masks_from_jsonl(
    jsonl_path="output_masks.jsonl",
    input_folder="your_image_folder",
    output_folder="visualized_masks",
    alpha=0.3,
    max_images=10  # Optional
)
```

This overlays masks on images and saves them to an output folder.

---

## Output JSONL Format

Each line in the output file looks like:

```json
{
  "image_id": "example.jpg",
  "masks": [
    {
      "bbox": [x, y, w, h],
      "area": 1234.56,
      "score": 0.97,
      "rle": {
        "size": [height, width],
        "counts": "e1a5Z0..."
      }
    }
  ]
}
```

- `rle`: COCO-compatible run-length encoding
- `score`: predicted IoU from SAM

---
