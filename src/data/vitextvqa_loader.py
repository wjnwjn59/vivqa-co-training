import json
import os
from torch.utils.data import Dataset
from src.utils.img_handler import load_image

class ViTextVQADataset(Dataset):
    def __init__(self, root_dir, subset="train"):
        self.root_dir = root_dir
        self.subset = subset
        self.image_dir = os.path.join(self.root_dir, "st_images")
        self.images_id_lst, self.annotation_lst = self.load_anntotations() 

    def load_anntotations(self):
        if self.subset == "train":
            data_path = os.path.join(self.root_dir, "ViTextVQA_train.json")
        elif self.subset == "dev":
            data_path = os.path.join(self.root_dir, "ViTextVQA_dev.json")
        elif self.subset == "test":
            data_path = os.path.join(self.root_dir, "ViTextVQA_test.json")
        else:
            raise ValueError("Subset must be one of 'train', 'test', or 'dev'.")
    
        data = json.load(open(data_path, 'r'))
        images_id_lst, annotation_lst = data["images"], data["annotations"]

        return images_id_lst, annotation_lst

    def __len__(self):
        return len(self.annotation_lst)

    
    def __getitem__(self, idx):
        item = self.annotation_lst[idx]
        sample_id = item["id"]
        image_id = item["image_id"]
        question = item["question"]
        answers_lst = item["answers"]

        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image_pil = load_image(image_path)

        item_dict = {
            "sample_id": sample_id,
            "image": image_pil,
            "question": question,
            "answers": answers_lst,
        }

        return item_dict
    
if __name__ == "__main__":
    dataset = ViTextVQADataset(root_dir="/mnt/VLAI_data/ViTextVQA", subset="train")
    print(f"Number of samples in the dataset: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answers']}")
    print(f"Image size: {sample['image'].size}")  