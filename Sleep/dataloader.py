from conf import *

import os
import cv2
import copy
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.functional import img_to_tensor
from tqdm import tqdm


class SleepDataset(Dataset):
    def __init__(self, args, image_paths, meta_data=None, labels=None, transforms=None, use_masking=False, is_test=False):
        self.args = args
        self.image_paths = image_paths
        self.meta = meta_data
        self.use_meta = args.use_meta
        self.use_masking = use_masking
        self.labels = labels
        self.images = []
        self.transforms = transforms
        self.is_test = is_test
        
        print('########################### dataset loader')
        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, -1)
            self.images.append(image)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transforms:
            image = self.transforms(image=image)['image']

        if self.use_masking:
            mask_num = random.randrange(1, args.max_mask_num+1)
            if self.args.masking_type == 'soft':
                mask_prob = 0.25
                rand_val = random.random()
                if rand_val < mask_prob:
                    image = time_mask(image, num_masks=mask_num)
                rand_val = random.random()
                if rand_val < mask_prob:
                    image = signal_mask(image, num_masks=mask_num)

            elif self.args.masking_type == 'hard':
                mask_prob = 0.3
                rand_val = random.random()
                if rand_val < mask_prob:
                    image = signal_mask(time_mask(image, num_masks=mask_num), num_masks=mask_num)
                else:
                    rand_val = random.random()
                    if rand_val < 0.3:
                        image = time_mask(image, num_masks=mask_num)
                    rand_val = random.random()
                    if rand_val < 0.3:
                        image = signal_mask(image, num_masks=mask_num)

        data = {}
        if self.use_meta:
            data['image'] = img_to_tensor(image)
            data['meta'] = torch.tensor(self.meta[index]).float()
        else:
            data['image'] = img_to_tensor(image)

        if self.is_test:
            return data
        else:
            label = self.labels[index]
            return data, label


def padding_img(img):
    BLOCK = [0]
    return cv2.copyMakeBorder(img.copy(),105,105,0,0,cv2.BORDER_CONSTANT,value=BLOCK)


def time_mask(image, T=30, num_masks=1):
    cloned = copy.deepcopy(image)
    len_spectro = cloned.shape[1]

    for i in range(0, num_masks):
        t = random.randrange(10, T)
        t_zero = random.randrange(10, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): 
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        cloned[:,t_zero:mask_end] = 0
    
    return cloned


def signal_mask(image, S=30, num_masks=1):
    cloned = copy.deepcopy(image)
    num_mel_channels = cloned.shape[1]
    
    for i in range(0, num_masks):        
        f = random.randrange(10, S)
        f_zero = random.randrange(10, num_mel_channels - f)
        
        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): 
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f) 
        cloned[f_zero:mask_end] = 0
    
    return cloned