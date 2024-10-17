import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import rotate
import glob

class KvasirDataset(Dataset):
    def __init__(self, data_path , transform_img = None, transform_mask = None):
        self.data_path = data_path
        self.image_list = glob.glob(self.data_path+'/images/*.png')
        self.label_list = glob.glob(self.data_path+'/masks/*.png')

        self.transform_img = transform_img
        self.tranform_mask = transform_mask

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """Get the images"""
        image_path = self.image_list[index]
        msk_path = self.label_list[index]

        img = Image.open(image_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        name = self.image_list[index].split('/')[-1].split('.')[0]
        if self.transform_img:
            state = torch.get_rng_state()
            img = self.transform_img(img)
            torch.set_rng_state(state)
            mask =  self.tranform_mask(mask)
        return (img, mask, name)