import torch
import clip
import requests
from utils import refine_classname, get_imagenet_classes
from utils import clamp

def clip_img_preprocessing(X, device):
    X = torch.nn.functional.upsample(X, size=(224, 224), mode='bicubic')
    IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
    mu = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)
    X = (X - mu) / std
    return X

def clip_text_emb(model, Imagenet = True, text_tokens = None, device = 'cuda'):
    text_emb_total = []
    if Imagenet:
        class_idx, imagenet_classes = get_imagenet_classes()
        imagenet_classes = refine_classname(imagenet_classes)
        template = "This is a photo of a {}"
        text = [template.format(c) for c in imagenet_classes]
        text_tokens = clip.tokenize(text).to(device)
        for j in range(len(text_tokens)):
            text_emb = model.encode_text(text_tokens[j].unsqueeze(0)).to('cpu')
            text_emb_total.append(text_emb)
        text_emb_total = torch.cat(text_emb_total, dim=0)
    return text_emb_total

def clip_output(model, images, text_emb):
    images_emb = model.encode_image(images)
    logits_per_image = images_emb @ text_emb.t()
    logits_per_text = text_emb @ images_emb.t()    
    return logits_per_image, logits_per_text

        