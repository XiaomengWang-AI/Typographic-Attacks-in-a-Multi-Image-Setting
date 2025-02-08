import os
import random
import shutil
from build_imagenet_T_dataset import build_synsets_dict
from utils import get_imagenet_classes
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import cv2
from utils import put_text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval().to(device)  
tokenizer = open_clip.get_tokenizer('ViT-B-32')

class_idx, imagenet_classes = get_imagenet_classes()
text = torch.cat([tokenizer(f"a photo of a {c}") for c in imagenet_classes]).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)


synsets_dict, _ = build_synsets_dict()
directory_path = 'correct_imagenet_dataset/'
folders = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
image_extensions = ['.JPEG']
folders_with_images = [
    folder for folder in folders 
    if any(file.name.endswith(tuple(image_extensions)) for file in os.scandir(os.path.join(directory_path,folder)))
]


labels = pd.read_csv('./data/1gram_labels_imagenet.csv', delimiter=',')
moderately_attack_text_list = labels['ori_label'].tolist()

global typo_synset
typo_synset_list = []
for i in tqdm(range(len(moderately_attack_text_list))):
    typo_text = moderately_attack_text_list[i]
    if moderately_attack_text_list[i] in labels['class_label'].tolist():
        typo_synset = labels[labels['class_label'] == moderately_attack_text_list[i]]['class_id'].values[0]
        typo_synset_list.append(typo_synset)


for i in tqdm(range(len(moderately_attack_text_list))):
    typo_text = moderately_attack_text_list[i]
    if moderately_attack_text_list[i] in labels['class_label'].tolist():
        typo_synset = labels[labels['class_label'] == moderately_attack_text_list[i]]['class_id'].values[0]
        if typo_synset in folders_with_images:
            folders_with_images.remove(typo_synset)

    typo_id = synsets_dict[typo_synset][0]

    results = []
    targeted_num = 0
    untargeted_num = 0
    for synset_path in folders_with_images:
        golden_id = synsets_dict[synset_path][0]  
        # golden_label = synsets_dict[synset_path][1]
        # target_label = class_idx[str(golden_id)][1]
        full_synset_path = os.path.join(directory_path, synset_path)
        image_path_li = os.listdir(full_synset_path)
        for image_path in image_path_li:
            attack_flag = 2
            full_image_path = os.path.join(full_synset_path, image_path)
            image = preprocess(Image.open(full_image_path)).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image).to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                typo_text_features = model.encode_text(tokenizer(["a photo of a " + typo_text]).to(device))
                typo_text_features /= typo_text_features.norm(dim=-1, keepdim=True)
            typo_text_image_similarity = torch.nn.functional.cosine_similarity(image_features, typo_text_features)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            class_prob = text_probs[0][golden_id].item()
            img = cv2.imread(full_image_path)
            typo_img = img.copy()
            flag = put_text(typo_img, typo_text, color=(0,255,255))
            # cv2.imwrite('test.jpg',typo_img)
            if flag:
                typo_img = Image.fromarray(typo_img)
                typo_img_tensor = preprocess(typo_img).unsqueeze(0).to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    typo_img_features = model.encode_image(typo_img_tensor).to(device)
                    typo_img_features /= typo_img_features.norm(dim=-1, keepdim=True)
                    output = (100.0 * typo_img_features @ text_features.T).softmax(dim=-1)  
                
                typo_prob, typo_indices = torch.max(output, dim=1)  
                indices = typo_indices.item()
                if indices != golden_id:
                    attack_flag = 0
                    if indices == typo_id:
                        attack_flag = 1
                    
            output_string = f'{image_path},{golden_id},{synset_path},{class_prob},{typo_text_image_similarity.item()},{class_idx[str(indices)][1]},{attack_flag}'
            results.append(output_string)


    with open('', 'a') as f:
        f.write('image_path,id,synset,visual_prob,similarity,predict_label,attack_flag\n')
        for i in range(len(results)):
            f.write(results[i] + '\n')


