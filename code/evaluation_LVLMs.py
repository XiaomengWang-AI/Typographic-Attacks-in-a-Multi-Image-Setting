import torch
from PIL import Image
import pandas as pd 
import numpy as np
import os
from tqdm import tqdm
import random
import psutil
import gc
from torch.utils.data import Dataset, DataLoader
from utils_models import *
from utils import get_imagenet_classes


# 自定义 collate_fn
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    golden_ids, image_paths, images, attack_texts = zip(*batch)
    images = [torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32) / 255.0 for img in images]
    return golden_ids, image_paths, torch.stack(images), attack_texts

# Dataset 类
class ImageDataset(Dataset):
    def __init__(self, dataset_path, df):
        self.dataset_path = dataset_path
        self.df = df
        self.image_list = []
        for golden_id in os.listdir(dataset_path):
            image_paths = os.listdir(os.path.join(dataset_path, golden_id))
            for image_path in image_paths:
                self.image_list.append((golden_id, image_path))
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        golden_id, image_path = self.image_list[idx]
        full_image_path = os.path.join(self.dataset_path, golden_id, image_path).replace("\\", "/")
        relative_image_path = os.path.join(golden_id, image_path).replace("\\", "/")

        filtered_df = self.df[self.df['image_path'] == relative_image_path]
        if filtered_df.empty:
            print(f"No match found for image path: {relative_image_path}")
            print(f"Expected Path: {self.df['image_path']}")
            return None
        

        attack_text = filtered_df['attack_text'].values[0]
        image = Image.open(full_image_path).convert("RGB").resize((224, 224))
        return golden_id, image_path, image, attack_text


def process_images(model, df, dataset_path, output_file):
    class_idx, imagenet_classes = get_imagenet_classes()

    dataset = ImageDataset(dataset_path, df)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    results = []
    none_batch_count = 0

    for batch in tqdm(dataloader):
        if batch is None:
            none_batch_count += 1
            continue
        
        golden_id, image_path, images, attack_text = batch
        golden_label = next((value[1] for key, value in class_idx.items() if value[0] == golden_id[0]), None)
        golden_label = golden_label.lower().strip().replace("_", " ").replace("-", " ")
        attack_text = [x.lower().strip().replace("_", " ").replace("-", "") for x in attack_text]
        prompt, golden_index, golden_answer = set_typoprompt_two(golden_label,attack_text[0])
        print(f"Prompt: {prompt}")

        with torch.no_grad():
            if model["name"] == 'blip':
                text_output = run_blip(prompt, [images[0]], model, device)
            elif model["name"] == "llava":
                text_output = run_llava(prompt, [images[0]], model, device)
            elif model["name"] == "minigpt4":
                text_output = run_minigpt4(prompt, [images[0]], model)
            elif model["name"] == "gpt4":
                gpt4_image_path = os.path.join(dataset_path, golden_id[0], image_path[0])
                text_output = run_gpt4(gpt4_image_path, prompt)
               
        if text_output is None:
            continue

        text_output = [x.replace(",", " ").replace("\n", " ") for x in text_output]
        image_name = os.path.join(golden_id[0], image_path[0])
        results.append([image_name, text_output[0], golden_label, attack_text[0], golden_answer, golden_index])

        del images
        del text_output
        torch.cuda.empty_cache()
        gc.collect()

    with open(output_file, "w") as f:
        f.write("image_path,text_output,golden_label,attack_text,golden_answer,golden_index\n")
        for result in results:
            f.write(",".join([str(x) for x in result]) + "\n")
    
    print(f"Processing complete. Results saved to {output_file}.")
    print(f"Skipped {none_batch_count} unmatched images.")


if __name__ == "__main__":
    # 初始化
    torch.cuda.init()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_data = get_model_data("blip")
    model_name = model_data["name"]
    for i in range (1, 6):
        # load the attack text csv file
        df = pd.read_csv('')
        # input the evaluation dataset path
        dataset_path = ''
        output_file = ''

        process_images(model_data, df, dataset_path, output_file)       
	





