import json
import os
from torch.utils.data import Dataset
from src.utils.img_handler import load_image

class OpenViVQADataset(Dataset):
    def __init__(self, root_dir, subset="train"):
        self.root_dir = root_dir
        self.subset = subset
        self.image_dir = self._get_image_dir()
        self.images_id_lst, self.annotation_kv_lst = self._load_anntotations() # error in images id lst mapping

    def _get_image_dir(self):
        if self.subset == "train":
            return os.path.join(self.root_dir, "training-images")
        elif self.subset == "dev":
            return os.path.join(self.root_dir, "dev-images")
        elif self.subset == "test":
            return os.path.join(self.root_dir, "test-images")
        else:
            raise ValueError("Subset must be one of 'train', 'test', or 'dev'.")

    def _load_anntotations(self):
        if self.subset == "train":
            data_path = os.path.join(self.root_dir, "vlsp2023_train_data.json")
        elif self.subset == "dev":
            data_path = os.path.join(self.root_dir, "vlsp2023_dev_data.json")
        elif self.subset == "test":
            data_path = os.path.join(self.root_dir, "vlsp2023_test_data.json")
        else:
            raise ValueError("Subset must be one of 'train', 'test', or 'dev'.")
    
        data = json.load(open(data_path, 'r'))
        images_id_dict, annotation_kv_lst = data["images"], list(data["annotations"].items())

        return images_id_dict, annotation_kv_lst

    def __len__(self):
        return len(self.annotation_kv_lst)

    
    def __getitem__(self, idx):
        item = self.annotation_kv_lst[idx]
        sample_id = item[0]
        sample = item[1]

        question = sample["question"]
        answer = sample["answer"]
        image_id = sample["image_id"]
        image_path = os.path.join(self.image_dir, f"{image_id:012d}.jpg")
        image_pil = load_image(image_path)

        item_dict = {
            "sample_id": sample_id,
            "image": image_pil,
            "question": question,
            "answers": [answer],
        }

        return item_dict
    

if __name__ == "__main__":
    dataset = OpenViVQADataset(root_dir="/mnt/VLAI_data/OpenViVQA", subset="train")
    print(f"Number of samples in the dataset: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answers']}")
    print(f"Image size: {sample['image'].size}")