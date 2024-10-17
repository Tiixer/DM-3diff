import torch
import torch.nn
import numpy as np
import os
import os.path
import h5py
import torchvision.transforms.functional as TF

class BRATSDataset_5H(torch.utils.data.Dataset):
    def __init__(self,data_path ,transform = None, test_flag=False):
        self.directory = os.path.expanduser(data_path)
        self.data_list = os.listdir(data_path)
        self.transform = transform
        self.test_flag = test_flag
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        """
        Get the images
        """
        name = self.data_list[index].split('.')[0]
        file_path = os.path.join(self.directory, self.data_list[index])
        with h5py.File(file_path, 'r') as f:
            t1 = f['t1'][:]
            t2 = f['t2'][:]
            t1ce = f['t1ce'][:]
            flair = f['flair'][:]
            img = torch.tensor(np.stack([t1,t2,t1ce,flair]),dtype=torch.float32)
            mask = torch.tensor(f['seg'][:],dtype=torch.float32).unsqueeze(0)
        if self.test_flag:
            if self.transform:
                return (self.transform((img, mask, name)))
        else:
            if self.transform:
                state = torch.get_rng_state()
                img, mask, name = self.transform((img, mask, name))
                torch.set_rng_state(state)
                return (img, mask, name)
        return (img, mask, name)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask , name = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w]), name
    

class myNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
            
    def __call__(self, data):
        img, msk, name = data
        img_normalized = TF.normalize(img, self.mean, self.std)
        img_normalized = ((img_normalized - img_normalized.min()) 
                            / (img_normalized.max() - img_normalized.min()))
        return img_normalized, msk, name