import numpy as np
import cv2
import pickle
import os
from tqdm import tqdm
import scipy.io as sio



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



