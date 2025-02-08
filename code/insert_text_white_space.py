from PIL import Image, ImageDraw, ImageFont
import textwrap
import pandas as pd
import os
import numpy as np
from utils import get_imagenet_classes
import torch
import random
from tqdm import tqdm


def add_white_background(image): 
	image_width, image_height = image.size
	white_height = int(image_height * 0.25)
	new_image = Image.new('RGB', (image_width, image_height + white_height*2), 'white')
	new_image.paste(image, (0, white_height))
	return new_image, image_height


def insert_text(image, text, image_height_old, font_path = "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf", font_size = 20, color = "black", position="top"): 

	image_width, image_height = image.size

	# Set up the font
	try:
		font = ImageFont.truetype(font_path, font_size)
	except IOError:
		font = ImageFont.load_default()

	# Prepare text wrapping
	lines = textwrap.wrap(text, width=int(image_width / (font_size * 0.45)))

	# Initialize drawing context
	draw = ImageDraw.Draw(image)

	# Write text
	# y_text = image_height_old
	y_text = 0 if position == "top" else image_height_old + int(image_height_old * 0.25)
	for line in lines:
		line_width, line_height = font.getsize(line)
		draw.text((0, y_text), line, font=font, fill=color)
		y_text += line_height + 5

	return image


class_idx, imagenet_classes = get_imagenet_classes()
method = 'random'
for i in range(1, 6):
    dataset_path = f''
    classes = os.listdir(dataset_path)
    # load the attack text csv file
    df = pd.read_csv(f'')
    topattack_image_path = f''
    for golden_id in tqdm(classes):
        images = os.listdir(os.path.join(dataset_path, golden_id))
        golden_label = next((value[1] for key, value in class_idx.items() if value[0] == golden_id), None)
        golden_label = golden_label.lower().replace('_', ' ').replace('-', ' ')
        for image in images:
            image_path = os.path.join(golden_id, image)
            image = Image.open(os.path.join(dataset_path, golden_id, image)).convert("RGB")
            width, height = image.size
            #  add the white background
            image = np.array(image)
            image = Image.fromarray(image).convert("RGB")
            image, image_height = add_white_background(image)

            find the data with the same image_path in df
            filtered_df = df[df['image_path'] == image_path]['attack_text']
            if not filtered_df.empty:
                attack_text = filtered_df.values[0]
            # Proceed with using attack_text
            else:
                print(f"No entry found for image path: {image_path}")
            new_image = insert_text(image, attack_text, image_height, color="black", font_size = 20, position="top")
            new_image = add_text_onimage(image, attack_text, image_height=height, image_width=width)
            new_image_path = os.path.join(topattack_image_path, image_path)
            if not os.path.exists(os.path.dirname(new_image_path)):
                os.makedirs(os.path.dirname(new_image_path))
            new_image.save(new_image_path)

