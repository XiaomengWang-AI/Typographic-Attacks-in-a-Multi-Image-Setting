from tqdm import tqdm
import torch
import os
from PIL import Image
import open_clip
from open_clip import tokenizer
import scipy.io as sio
from utils import get_imagenet_classes
import cv2
from build_imagenet_T_dataset import put_text
import pandas as pd

def build_synsets_dict():
    synsets_info_path = '../ILSVRC2012_devkit_t12/data/meta.mat'
    raw_data = sio.loadmat(synsets_info_path)['synsets']

    synsets_list = []
    reverse_dict = {}

    for i, synset in enumerate(raw_data):
        if i == 1000:
            break
        orig_synset = synset[0]

        synset_id = orig_synset[1].item()
        synset_label = orig_synset[2].item()
        gloss = orig_synset[3].item()
        synsets_list.append((synset_id, synset_label, gloss))

    sorted_list = sorted(synsets_list, key=lambda item: item[0])

    synsets_dict = {}
    for i, (synset_id, synset_label, gloss) in enumerate(sorted_list):
        synsets_dict[synset_id] = (i, synset_label, gloss)
        reverse_dict[i] = synset_id

    return synsets_dict, reverse_dict

def build_dir(base_path):
    synsets_dict, reverse_dict = build_synsets_dict()
    os.mkdir(base_path)
    for synset_id, _ in synsets_dict.items():
        construct_path = os.path.join(base_path, synset_id)
        os.mkdir(construct_path)

build_dir('correct_imagenet_dataset')

def get_imageNet_synsets_info():
    synsets_info_path = '../data/meta.mat'
    raw_data = sio.loadmat(synsets_info_path)['synsets']

    synsets_list = []

    for i, synset in enumerate(raw_data):
        if i == 1000:
            break
        orig_synset = synset[0]

        synset_id = orig_synset[1].item()
        synset_label = orig_synset[2].item()
        gloss = orig_synset[3].item()
        synsets_list.append((synset_id, synset_label, gloss))

    sorted_list = sorted(synsets_list, key=lambda item: item[0])
    synsets_dict = {}
    for i, (synset_id, synset_label, gloss) in enumerate(sorted_list):
        synsets_dict[synset_id] = (i, synset_label, gloss)

    return synsets_dict

synsets_dict = get_imageNet_synsets_info()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model,_, preprocess = open_clip.create_model_and_transforms(model_name='ViT-B-32',pretrained='laion2b_s34b_b79k', device=device)
model.eval().to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
class_idx, imagenet_classes = get_imagenet_classes()
text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in imagenet_classes]).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

base_dir = ''
catetory_li = os.listdir(base_dir)

df = pd.read_csv('.data/1gram_labels_imagenet.csv', delimiter=',')
attack_texts = df['class_label'].tolist()
class_id = df['key'].tolist()

for i, attack_text in tqdm(enumerate(attack_texts)):
# total_correct = 0
    targeted_num = 0
    untargeted_num = 0   
    total = 0
    for catetory_id in catetory_li:       
        class_correct = 0
        (golden_id, synset_label, gloss) = synsets_dict[catetory_id]
        all_image_path = os.path.join(base_dir, catetory_id)
        all_images_li = os.listdir(all_image_path)
        class_total = len(all_images_li)
        total = total + len(all_images_li)
        for image_path in all_images_li:      
            full_image_path = os.path.join(all_image_path, image_path)
            img = cv2.imread(full_image_path)
            typo_img = img.copy()
            flag = put_text(typo_img, attack_text, color=(0,255,255))
            if flag:
                typo_img = Image.fromarray(typo_img)
                typo_img_tensor = preprocess(typo_img).unsqueeze(0).to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(typo_img_tensor)  # 1, 512
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    output = (100.0 * image_features @ text_features.T).softmax(dim=-1)  
                
                _, indices = torch.max(output, dim=1)  # shape: 1
                indices = indices.item()
                if indices != golden_id:
                    untargeted_num += 1
                if indices == int(class_id[i]):
                    targeted_num += 1
                    
    with open('', 'a') as f:
        f.write(f'{attack_text},{total},{untargeted_num},{untargeted_num/total},{targeted_num},{targeted_num/total}\n')






