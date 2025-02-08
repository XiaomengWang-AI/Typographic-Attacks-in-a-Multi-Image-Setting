import shutil
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import Dataset
import cv2
import numpy as np


def get_imagenet_classes():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    class_idx = requests.get(url).json()
    imagenet_classes = [class_idx[str(i)][1] for i in range(1000)]
    return class_idx, imagenet_classes

def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names

def clamp(images, lower_limit, upper_limit):
    return torch.max(torch.min(images, upper_limit), lower_limit)   

count = 0
def put_text(image, text, color):
    def check_point(target_x, target_y):
        def check_single_point(x, y):
            for point in points_li:
                if x > point[0] and x < point[0] + width and y < point[1] and y > point[1] - height:
                    return False
            return True
        return check_single_point(target_x, target_y) & check_single_point(target_x+width, target_y) \
        & check_single_point(target_x, target_y-height) & check_single_point(target_x+width, target_y-height)
    m = image.shape[1]
    n = image.shape[0]
    global count
    (width, height), baseLine = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=2)
    if m-width <= 0 or height >= n:
        count += 1
        return False
    points_li = []
    for i in range(8):
        while True:
            x = np.random.randint(0, m - width)
            y = np.random.randint(0 + height, n)
            if check_point(x,y):
                break

        points_li.append((x,y))
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
    return True

