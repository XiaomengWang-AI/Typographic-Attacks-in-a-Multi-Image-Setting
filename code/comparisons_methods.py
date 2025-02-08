import os
import random
from pathlib import Path
import shutil
import pandas as pd
import torch
import open_clip
from utils import get_imagenet_classes
import cv2
from PIL import Image
import numpy as np
from utils import put_text
from tqdm import tqdm


## general model setting and text encoding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model,_, preprocess = open_clip.create_model_and_transforms(model_name='ViT-B-32',pretrained='laion2b_s34b_b79k', device=device)
model.eval().to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
class_idx, imagenet_classes = get_imagenet_classes()
text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in imagenet_classes]).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

results = []
for i in range(1, 6):
    directory_path = 'correct_imagenet_dataset'
    destination_directory = 'random_579class_images'
    os.mkdir(destination_directory)
    folders = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    image_extensions = ['.JPEG']
    folders_with_images = [
        folder for folder in folders 
        if any(file.name.endswith(tuple(image_extensions)) for file in os.scandir(os.path.join(directory_path,folder)))
    ]
    selected_folders = random.sample(folders_with_images, 579)
    selected_images = {}
    for folder in selected_folders:
        images = [file.path for file in os.scandir(os.path.join(directory_path,folder)) if file.name.endswith(tuple(image_extensions))]
        selected_images[folder] = random.choice(images)
    for folder, image in selected_images.items():
        class_destination_directory = os.path.join(destination_directory, folder)
        os.mkdir(class_destination_directory)
        shutil.copy(image, class_destination_directory)

    folders = [d for d in os.listdir(destination_directory)]
    images = [os.path.join(folders[i],img) for i in range(len(folders)) for img in os.listdir(os.path.join(destination_directory, folders[i]))]


# # image in vid order and match by similarity
    attack_texts = pd.read_csv('./data/1gram_labels_imagenet.csv', delimiter=',')['class_label'].tolist()
    attack_texts_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in attack_texts]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        attack_texts_features = model.encode_text(attack_texts_inputs)
        attack_texts_features /= attack_texts_features.norm(dim=-1, keepdim=True)
    img_prob_dict = {}
    untargeted_num = 0
    targeted_num = 0
    max_index_list = []
    img_prob_dict = {}
    for img in images:
        synset_path = img.split('/')[0]
        image_path = os.path.join(destination_directory, img)
        ori_image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            ori_image_features = model.encode_image(ori_image_tensor)
            ori_image_features /= ori_image_features.norm(dim=-1, keepdim=True)
            ori_output = (100.0 * ori_image_features @ text_features.T).softmax(dim=-1) 
            golden_prob, golden_id = torch.max(ori_output, dim=1)
            # sort the image by the image prediction probability
            img_prob_dict[img] = [golden_id.item(),golden_prob.item()]
            # img_prob_dict[img] = [golden_id.item(), df[df['synset'] == synset_path]['visual_prob'].values[0]]

    # sort the dictionary by the prediction probability in ascending order
    img_prob_dict = dict(sorted(img_prob_dict.items(), key=lambda item: item[1][1], reverse=False))
    # only get the image
    images = list(img_prob_dict.keys())

    for img in tqdm(images):
        image_path = os.path.join(destination_directory, img)
        ori_image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            ori_image_features = model.encode_image(ori_image_tensor)
            ori_image_features /= ori_image_features.norm(dim=-1, keepdim=True)
            ori_output = (100.0 * ori_image_features @ text_features.T).softmax(dim=-1) 
            golden_prob, golden_id = torch.max(ori_output, dim=1)
            golden_label = imagenet_classes[golden_id.item()]
            if golden_label in attack_texts:
                attack_texts.remove(golden_label)
            attack_texts_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in attack_texts]).to(device)
            attack_texts_features = model.encode_text(attack_texts_inputs)
            attack_texts_features /= attack_texts_features.norm(dim=-1, keepdim=True)
            # compute the similarity between the image and the attack text
            similarity = torch.nn.functional.cosine_similarity(ori_image_features, attack_texts_features)
            # get the attack text with the highest similarity
            max_value = torch.max(similarity)
            max_index = torch.argmax(similarity)
            # # if there is no same index with max_index in max_index_list, then add the attack text to the image
            if max_index.item() not in max_index_list:
                max_index_list.append(max_index.item())
            else:
                max_index = torch.argsort(similarity, descending=True)[1]

        img = cv2.imread(image_path)
        typo_img = img.copy()
        flag = put_text(typo_img, attack_texts[max_index.item()], color=(0,255,255))
        # cv2.imwrite('test.jpg',typo_img)
        if flag:
            typo_img = Image.fromarray(typo_img)
            typo_img_tensor = preprocess(typo_img).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(typo_img_tensor)  
                image_features /= image_features.norm(dim=-1, keepdim=True)
                output = (100.0 * image_features @ text_features.T).softmax(dim=-1)  
            
            _, indices = torch.max(output, dim=1) 
            if indices != golden_id:
                untargeted_num += 1
            if class_idx[str(indices.item())][1] == attack_texts[max_index.item()]:
                targeted_num += 1
    output_string = f'{i},{untargeted_num},{untargeted_num/len(images)},{targeted_num},{targeted_num/len(images)}'
    results.append(output_string)
    shutil.rmtree(destination_directory)




# use the most effective attack text (upper bound) 'groenendael'
    # untargeted_num = 0
    # targeted_num = 0
    # for img in images:
    #     image_path = os.path.join(destination_directory, img)
    #     ori_image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    #     with torch.no_grad(), torch.cuda.amp.autocast():
    #         ori_image_features = model.encode_image(ori_image_tensor)
    #         ori_image_features /= ori_image_features.norm(dim=-1, keepdim=True)
    #         ori_output = (100.0 * ori_image_features @ text_features.T).softmax(dim=-1)
    #     golden_prob, golden_id = torch.max(ori_output, dim=1)
    #     img = cv2.imread(image_path)
    #     typo_img = img.copy()
    #     flag = put_text(typo_img, 'groenendael', color=(0,255,255))
    #     # cv2.imwrite('test.jpg',typo_img)
    #     if flag:
    #         typo_img = Image.fromarray(typo_img)
    #         typo_img_tensor = preprocess(typo_img).unsqueeze(0).to(device)
    #         with torch.no_grad(), torch.cuda.amp.autocast():
    #             image_features = model.encode_image(typo_img_tensor)  # 1, 512
    #             image_features /= image_features.norm(dim=-1, keepdim=True)
    #             output = (100.0 * image_features @ text_features.T).softmax(dim=-1)  
    #       
    #         _, indices = torch.max(output, dim=1)  # shape: 1
    #         indices = indices.item()
    #         if indices != golden_id:
    #             untargeted_num += 1
    #         if indices == int('224'):
    #             targeted_num += 1




# images in descending order by visual_prob and attack texts in descending order by targeted_asr
    # dfid = pd.read_csv('./data/1gram_labels_imagenet.csv', delimiter=',')
    # img_prob_dict = {}
    # for img in images:
    #     image_path = os.path.join(destination_directory, img)
    #     ori_image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    #     with torch.no_grad(), torch.cuda.amp.autocast():
    #         ori_image_features = model.encode_image(ori_image_tensor)
    #         ori_image_features /= ori_image_features.norm(dim=-1, keepdim=True)
    #         ori_output = (100.0 * ori_image_features @ text_features.T).softmax(dim=-1)
    #     golden_prob, golden_id = torch.max(ori_output, dim=1)
    #     img_prob_dict[img] = [golden_id.item(), golden_prob.item()]
    # # sort the dictionary by the prediction probability in decending order
    # # img_prob_dict = dict(sorted(img_prob_dict.items(), key=lambda item: item[1][1], reverse=True))

    # df = pd.read_csv('./data//sorted_579label_8typo_yellow_pixeltext_asr.csv', delimiter=',')
    # # sorted by attack text effectiveness in descending order
    # df = df.sort_values(by='targeted_asr', ascending=False)
    # texts = df['ori_label'].tolist()
    # text_prob_dict = {}
    # for text in texts:
    #     targeted_asr = df[df['ori_label'] == text]['targeted_asr'].values[0]
    #     text_prob_dict[text] = targeted_asr

    # matched_pairs = list(zip(img_prob_dict.keys(), text_prob_dict.keys()))
    # untargeted_num = 0
    # targeted_num = 0
    # for img, attack_text in tqdm(matched_pairs):
    #     image_path = os.path.join(destination_directory, img)
    #     golden_id = img_prob_dict[img][0]
    #     img = cv2.imread(image_path)
    #     typo_img = img.copy()
    #     flag = put_text(typo_img, attack_text, color=(0,255,255))
    #     # cv2.imwrite('test.jpg',typo_img)
    #     if flag:
    #         typo_img = Image.fromarray(typo_img)
    #         typo_img_tensor = preprocess(typo_img).unsqueeze(0).to(device)
    #         with torch.no_grad(), torch.cuda.amp.autocast():
    #             image_features = model.encode_image(typo_img_tensor)  
    #             image_features /= image_features.norm(dim=-1, keepdim=True)
    #             output = (100.0 * image_features @ text_features.T).softmax(dim=-1)  
            
    #         _, indices = torch.max(output, dim=1) 
    #         indices = indices.item()
    #         if indices != golden_id:
    #             untargeted_num += 1
    #         if indices == dfid[dfid['class_label'] == attack_text]['key'].values[0]:
    #             targeted_num += 1
    # output_string = f'{i},{untargeted_num},{untargeted_num/len(images)},{targeted_num},{targeted_num/len(images)}'
    # results.append(output_string)
    # shutil.rmtree(destination_directory)




# random match
    # df = pd.read_csv('./data/1gram_labels_imagenet.csv', delimiter=',')
    # texts = df['class_label'].tolist()
    # random.shuffle(texts)
    # matched_pairs = list(zip(images, texts))
    # untargeted_num = 0
    # targeted_num = 0
    # for img, attack_text in tqdm(matched_pairs):
    #     attack_text_id = df[df['class_label'] == attack_text]['key'].values[0]
    #     image_path = os.path.join(destination_directory, img)
    #     ori_image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    #     with torch.no_grad(), torch.cuda.amp.autocast():
    #         ori_image_features = model.encode_image(ori_image_tensor)
    #         ori_image_features /= ori_image_features.norm(dim=-1, keepdim=True)
    #         ori_output = (100.0 * ori_image_features @ text_features.T).softmax(dim=-1)
    #     _, golden_id = torch.max(ori_output, dim=1)  

    #     img = cv2.imread(image_path)
    #     typo_img = img.copy()
    #     flag = put_text(typo_img, attack_text, color=(0,255,255))
    #     # cv2.imwrite('test.jpg',typo_img)
    #     if flag:
    #         typo_img = Image.fromarray(typo_img)
    #         typo_img_tensor = preprocess(typo_img).unsqueeze(0).to(device)
    #         with torch.no_grad(), torch.cuda.amp.autocast():
    #             image_features = model.encode_image(typo_img_tensor)  
    #             image_features /= image_features.norm(dim=-1, keepdim=True)
    #             output = (100.0 * image_features @ text_features.T).softmax(dim=-1)  
            
    #         _, indices = torch.max(output, dim=1)  
    #         indices = indices.item()
    #         if indices != golden_id:
    #             untargeted_num += 1
    #         if indices == attack_text_id:
    #             targeted_num += 1
    # output_string = f'{i},{untargeted_num},{untargeted_num/len(images)},{targeted_num},{targeted_num/len(images)}'
    # results.append(output_string)
    # shutil.rmtree(destination_directory)


# image with the highest visual_prob match attack text with lowest targeted_asr
# images in descending order by visual_prob and attack texts in ascending order by targeted_asr
    # dfid = pd.read_csv('./data/1gram_labels_imagenet.csv', delimiter=',')
    # img_prob_dict = {}
    # for img in images:
    #     image_path = os.path.join(destination_directory, img)
    #     ori_image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    #     with torch.no_grad(), torch.cuda.amp.autocast():
    #         ori_image_features = model.encode_image(ori_image_tensor)
    #         ori_image_features /= ori_image_features.norm(dim=-1, keepdim=True)
    #         ori_output = (100.0 * ori_image_features @ text_features.T).softmax(dim=-1)
    #     golden_prob, golden_id = torch.max(ori_output, dim=1)
    #     img_prob_dict[img] = [golden_id.item(), golden_prob.item()]
    # # sort images by the prediction probability in decending order
    # img_prob_dict = dict(sorted(img_prob_dict.items(), key=lambda item: item[1][1], reverse=True))

    # df = pd.read_csv('./data/sorted_579label_8typo_yellow_pixeltext_asr.csv', delimiter=',')
    # # sorted by attack text effectiveness in descending order
    # df = df.sort_values(by='targeted_asr', ascending=True)
    # texts = df['ori_label'].tolist()
    # text_prob_dict = {}
    # for text in texts:
    #     targeted_asr = df[df['ori_label'] == text]['targeted_asr'].values[0]
    #     text_prob_dict[text] = targeted_asr

    # matched_pairs = list(zip(img_prob_dict.keys(), text_prob_dict.keys()))
    # untargeted_num = 0
    # targeted_num = 0
    # for img, attack_text in tqdm(matched_pairs):
    #     image_path = os.path.join(destination_directory, img)
    #     golden_id = img_prob_dict[img][0]
    #     img = cv2.imread(image_path)
    #     typo_img = img.copy()
    #     flag = put_text(typo_img, attack_text, color=(0,255,255))
    #     # cv2.imwrite('test.jpg',typo_img)
    #     if flag:
    #         typo_img = Image.fromarray(typo_img)
    #         typo_img_tensor = preprocess(typo_img).unsqueeze(0).to(device)
    #         with torch.no_grad(), torch.cuda.amp.autocast():
    #             image_features = model.encode_image(typo_img_tensor)  
    #             image_features /= image_features.norm(dim=-1, keepdim=True)
    #             output = (100.0 * image_features @ text_features.T).softmax(dim=-1)  
            
    #         _, indices = torch.max(output, dim=1) 
    #         indices = indices.item()
    #         if indices != golden_id:
    #             untargeted_num += 1
    #         if indices == dfid[dfid['class_label'] == attack_text]['key'].values[0]:
    #             targeted_num += 1
    # output_string = f'{i},{untargeted_num},{untargeted_num/len(images)},{targeted_num},{targeted_num/len(images)}'
    # results.append(output_string)
    # shutil.rmtree(destination_directory)



with open('', 'w') as f:
    f.write("i,untargeted_num,untargeted_asr,targeted_num,targeted_asr\n")  
    for item in results:
        f.write("%s\n" % item)
   
