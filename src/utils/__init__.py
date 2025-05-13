from PIL import Image


def load_image(image_path):
    """
    Load an image from a given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Loaded image.
    """
    return Image.open(image_path)