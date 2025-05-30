from PIL import Image

def load_image(image_path: str): 
    image = Image.open(image_path).convert('RGB')
    return image