import random
import torch
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import requests
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd 
import seaborn as sns

import cv2
import numpy as np
import os
from tqdm import tqdm
import string

import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import textwrap
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import transformers
from utils import get_imagenet_classes




def set_typoprompt_two(golden_label, attack_text):
	prompts, choices = [],[]
	options = {}
	choices.append(golden_label)
	choices.append(attack_text)
	random.shuffle(choices)
	question = f"Select the correct object pictured in the image: "
	for i, choice in enumerate(choices):
		options[f'{i+1}'] = choice
		question += f"({i+1}) {choice} or "
	question = question[:-4]
	question += ". Answer with either (1) or (2) only."
	prompts.append(question)
	golden_index = choices.index(golden_label) + 1
	answer = f"({golden_index}) {golden_label}"
	return prompts, golden_index, answer



def get_model_data(name): 
	if name == "blip":

		from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
		
		model_id = "Salesforce/instructblip-flan-t5-xxl"

		model = InstructBlipForConditionalGeneration.from_pretrained(
    	model_id,
    	device_map = 'auto',  # 自动映射到多个设备（GPU/CPU）
    	offload_folder="",  # 卸载到磁盘
   		torch_dtype=torch.float16,
		low_cpu_mem_usage=True  # 使用更小的浮点类型（降低内存占用）
		)
		processor = InstructBlipProcessor.from_pretrained(model_id)


		model_data = {
			"model":model, 
			"processor":processor,
			"name":"blip"
		}

	if name == "llava":

		from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, pipeline

		model_id = "llava-hf/llava-1.5-13b-hf"
		
		# model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
		model = LlavaForConditionalGeneration.from_pretrained(
    	model_id,
    	device_map='auto',
    	offload_folder="",  
   		torch_dtype=torch.float16,
		low_cpu_mem_usage=True 
		)
		processor = AutoProcessor.from_pretrained(model_id)
		processor.image_processor.do_center_cropq = False
		processor.image_processor.size = {"height": 336, "width": 336}

		model_data = {
			"model":model, 
			"processor":processor,
			"name":"llava"
		}

	if name == "minigpt4": 

		import sys
		sys.path.insert(0,'./models/MiniGPT')

		from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData
		from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
		from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

		from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
		from minigpt4.conversation.conversation import CONV_VISION_minigptv2
		from minigpt4.common.config import Config


		def list_of_str(arg):
			return list(map(str, arg.split(',')))

		class Args: 
			def __init__(self): 
				self.cfg_path = "path-to-minigpt4/MiniGPT/eval_configs/minigptv2_eval.yaml"
				self.name = 'A2'
				self.eval_opt = 'all'
				self.max_new_tokens = 10
				self.batch_size = 32
				self.lora_r = 64
				self.lora_alpha = 16 
				self.options=None

		args = Args() 
		cfg = Config(args)

		model, vis_processor = init_model(args)
		conv_temp = CONV_VISION_minigptv2.copy()
		conv_temp.system = ""
		model.eval()
		# save_path = cfg.run_cfg.save_path

		model_data = { 
			"model": model,  
			"name": "minigpt4", 
			"conv_temp":conv_temp,
			"vis_processor":vis_processor,
		}

	if name == "clip":
		import clip
		model, preprocess = clip.load("ViT-B/32")
		# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')	
		# tokenizer = open_clip.get_tokenizer('ViT-B-32')

		model_data = {
			"model":model, 
			"preprocess":preprocess,
			"name":"clip"
		}
	
	if name == "gpt4":
		model_data = {
			"name":"gpt4"
		}

	return model_data 



def run_blip(prompt, images, model_data, device, new_tokens = 50):

	inputs = model_data["processor"](images, text=prompt, return_tensors="pt", padding=True, truncation=True).to(device, dtype=torch.float16)

	outputs = model_data["model"].generate(
		**inputs,
		do_sample=False,
		max_new_tokens=new_tokens,
	).detach().cpu()

	text_output = model_data["processor"].batch_decode(outputs, skip_special_tokens=True)
	text_output = [text.strip().lower() for text in text_output]
	
	return text_output 


def run_llava(prompts, images, model_data, device, repeat=False,new_tokens = 50): 
	
    # prompts_fixed = [] 
    # for idx, prompt in enumerate(prompts): 
    #     prompts_fixed.append("USER: <image>\n" + prompt + "\nASSISTANT:")
	prompts_fixed = ["USER: <image>\n" + prompt + "\nASSISTANT:" for prompt in prompts]
	print(prompts_fixed)
# 	prompts_fixed = [
#     f"""USER: <image>
# {prompt}
# ASSISTANT:"""
#     for prompt in prompts
# 	]
	inputs = model_data["processor"](text = prompts_fixed, images=images, return_tensors="pt", padding=True).to(device, torch.float16)

	len_inputs_ids = [len(x) for x in inputs["input_ids"]]
	output = model_data["model"].generate(**inputs, max_new_tokens=new_tokens, do_sample=False,temperature=0.7)
	output = model_data["processor"].batch_decode(output, skip_special_tokens=True)  
	# print(output)      
	output = [text.split("\nASSISTANT:")[1].lower().strip() for text in output]

	return output



def run_minigpt4(prompt, image, model_data, new_tokens=20):
	from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
    

	model = model_data["model"]
	vis_processor = model_data["vis_processor"]
	conv_temp = model_data["conv_temp"]

	image = [vis_processor(image_).unsqueeze(0) for image_ in image] 
	image = torch.cat(image, dim=0)

	texts = prepare_texts(prompt, conv_temp)  
	answers = model.generate(image, texts, max_new_tokens=new_tokens)

	# print(len(answers))
	return answers


def run_gpt4(image_path, prompt):
	from utils_gpt4 import get_gpt4_pred
	answers = get_gpt4_pred(image_path, prompt)
	return answers