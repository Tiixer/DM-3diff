import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class BYYADataset(Dataset):
    def __init__(self, data_path , transform_img = None, transform_mask = None, mode = None):
        if mode == 'Training':
            df = pd.read_csv(os.path.join(data_path, 'BYYA+CAMPUS_Lv_' + mode + '_GroundTruth.csv'), encoding='gbk')
        elif mode == 'Testing':
            df = pd.read_csv(os.path.join(data_path, 'BYYA_Lv_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode

        self.transform_img = transform_img
        self.tranform_mask = transform_mask

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        
        if self.transform_img:
            state = torch.get_rng_state()
            img = self.transform_img(img)
            torch.set_rng_state(state)
            mask =  self.tranform_mask(mask)
        return (img, mask, name)