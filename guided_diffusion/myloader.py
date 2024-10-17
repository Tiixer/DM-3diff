import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate


class MyDataset(Dataset):
    def __init__(self, args, data_path, transform=None, mode='Training', plane=False):
        df = pd.read_csv(os.path.join(data_path, mode + '.csv'), encoding='gbk')
        self.name_list = df['name'].tolist()
        self.img_list = df['img_path'].tolist()
        self.label_list = df['mask_path'].tolist()
        self.data_path = data_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = self.img_list[index].replace('\\', '/')
        msk_path = self.label_list[index].replace('\\', '/')
        img = Image.open(os.path.join(self.data_path, img_path)).convert('RGB')
        mask = Image.open(os.path.join(self.data_path, msk_path)).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return (img, mask, name)
