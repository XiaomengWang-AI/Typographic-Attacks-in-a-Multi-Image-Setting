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


synsets_dict, reverse_dict = build_synsets_dict()
directory_path = 'correct_imagenet_dataset'
folders = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
image_extensions = ['.JPEG']
folders_with_images = [
    folder for folder in folders 
    if any(file.name.endswith(tuple(image_extensions)) for file in os.scandir(os.path.join(directory_path,folder)))
]


device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval().to(device)  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')
class_idx, imagenet_classes = get_imagenet_classes()
text = torch.cat([tokenizer(f"a photo of a {c}") for c in imagenet_classes]).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)


results = []
targeted_num = 0
untargeted_num = 0
for synset_path in tqdm(folders_with_images):
    golden_id = synsets_dict[synset_path][0]   #653
    # golden_label = synsets_dict[synset_path][1]
    target_label = class_idx[str(golden_id)][1]
    full_synset_path = os.path.join(directory_path, synset_path)
    image_path_li = os.listdir(full_synset_path)
    for image_path in image_path_li:
        attack_flag = 2
        full_image_path = os.path.join(full_synset_path, image_path)
        image = preprocess(Image.open(full_image_path)).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        class_prob = text_probs[0][golden_id].item()
        img = cv2.imread(full_image_path)
        typo_img = img.copy()
        typo_text = ''
        typo_id = ''
        flag = put_text(typo_img, typo_text, color=(0,255,255))

        if flag:
            typo_img = Image.fromarray(typo_img)
            typo_img_tensor = preprocess(typo_img).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(typo_img_tensor)  # 1, 512
                image_features /= image_features.norm(dim=-1, keepdim=True)
                output = (100.0 * image_features @ text_features.T).softmax(dim=-1)  
            
            typo_prob, typo_indices = torch.max(output, dim=1)  
            indices = typo_indices.item()
            if indices != golden_id:
                attack_flag = 0
                if indices == typo_id:
                    attack_flag = 1
        output_string = f'{image_path},{synset_path},{class_prob},{class_idx[str(indices)][1]},{attack_flag}'
        results.append(output_string)
        output_string = f'{image_path},{synset_path},{class_prob}'
        results.append(output_string)


with open('', 'a') as f:
    f.write('image_path,synset,visual_prob,predict_label,attack_flag\n')

