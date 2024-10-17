
import sys
import random
sys.path.append(".")
from guided_diffusion.utils import staple

import numpy
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import math
from PIL import Image
import matplotlib.pyplot as plt
from guided_diffusion.utils import staple
import argparse

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz


def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_seg(pred,true_mask_p,threshold = (0.1, 0.3, 0.5, 0.7, 0.9)):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--inp_pth")
    argParser.add_argument("--out_pth")
    args = argParser.parse_args()
    mix_res = (0,0)
    num = 0
    pred_path = args.inp_pth
    gt_path = args.out_pth
    ISIC2016_exclude = ["0011152","0011349","0011104","0010257","0011155","0011374","0008406","0000289","0010454","0010451","0003728","0010477","0009920","0010023","0010193","0000487","0000476","0000071","0009872","0011338","0010437","0000549","0011294","0003056","00010847","0000547","0000188","0000479","0010038"]
    ISIC2018_exclude = ["0036107","0021798","0015478","0015448","0021659","0036273","0023483","0036257","0036176","0024130","0023118","0015558","0022395","0022558","0023275","0036211","0036078","0036067","0036253","0036276","0021551","0023264","0036127","0024032","0015375","0023396","0036207","0012169","0024156","0019747","0021319","0023358","0022457","0036232","0021601","0023261","0036261","0021278","0036137","0021956","0022215","0020373","0022006","0021440","0017411","0024095","0023258","0015621","0036270","0021902","0015450","0021622","0021584","0021121","0036195","0022663","0022399","0036269","0023366","0015451","0023056","0024157","0023696","0015399","0036110","0022342","0022313","0019090","0023298","0017414","0022819","0036243","0023306","0021701","0036286","0021559","0023721","0017465","0021583","0021765","0023300","0020135","0023506","0012650","0036284","0024054","0022213","0023323","0036144","0036128","0019062","0036153","0022007","0022444","0021596","0021195","0023257","0024188","0023364","0023283","0022009","0036248","0021374","0023269","0022820","0023321","0021558","0022928","0036105","0036086","0021833","0036075","0021711""0021147""0021796"]
    BraTs_exclude = ["00732","01366","01502","00747","01495","01427","01554","01431","01075","01138","01154","01365","00514","00778","00607"]
    ClinicDB_exclude = ["52","349","575","425","21","205","73","154","251","106","559","14","25"]
    CVC_300_exclude = ["149","151","153","154","155","204","205","206","207","208","168"]

    for root, dirs, files in os.walk(pred_path, topdown=False):
        for name in files:
            # if 'ens' in name and name.split('_')[0] not in ISIC2016_exclude:
            # if 'ens' in name:
            # if name.split('.')[0].split('_')[1]:
            if name.split('_')[0] not in ClinicDB_exclude:
                num += 1
                ind = name.split('_')[0]
                # ind = name.split('_')[1]

                # ind = name.split('.')[0].split('_')[1]
                pred = Image.open(os.path.join(root, name)).convert('L')
                # gt_name = "BraTS2021_"+ind+'.png'
                # gt_name = 'Kvasir_' + ind + '.png'
                gt_name = ind+'.png'
                # gt_name = "ISIC_" + ind + "_Segmentation.png"
                # gt_name = name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2]+'.jpg'
                # gt_name = name.split('_')[1]+'.png'
                gt = Image.open(os.path.join(gt_path, gt_name)).convert('L')
                pred = torchvision.transforms.PILToTensor()(pred)
                pred = torchvision.transforms.Resize((256,256))(pred)
                pred = torch.unsqueeze(pred,0).float() 
                pred = pred / pred.max()
                # if args.debug:
                #     print('pred max is', pred.max())
                #     vutils.save_image(pred, fp = os.path.join('./results/' + str(ind)+'pred.jpg'), nrow = 1, padding = 10)
                gt = torchvision.transforms.PILToTensor()(gt)
                gt = torchvision.transforms.Resize((256,256))(gt)
                gt = torch.unsqueeze(gt,0).float() / 255.0
                # if args.debug:
                #     vutils.save_image(gt, fp = os.path.join('./results/' + str(ind)+'gt.jpg'), nrow = 1, padding = 10)
                temp = eval_seg(pred, gt)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
    iou, dice = tuple([a/num for a in mix_res])
    print('iou is',iou)
    print('dice is', dice)

if __name__ == "__main__":
    main()

# python scripts/segmentation_env.py --inp_pth Crop_RES/Images  --out_pth Croped_DATA/Test
# python scripts/segmentation_env.py --inp_pth Result/BRATS_RES/Images_DWT_V6 --out_pth Dataset/Processed_BraTs2021/test_mask
# python scripts/segmentation_env.py --inp_pth Result/ISIC_2018_RES/Images_DWT_V2  --out_pth Dataset/ISIC_2018/ISIC2018_Task1_Test_GroundTruth
# python scripts/segmentation_env.py --inp_pth Result/CVC_RES/CVC-ClinicDB_V2  --out_pth Dataset/CVC/CVC-ClinicDB/masks
# python scripts/segmentation_env.py --inp_pth Result/CVC_RES/CVC-ColonDB  --out_pth Dataset/CVC/CVC-ColonDB/masks
# python scripts/segmentation_env.py --inp_pth Result/CVC_RES/ETIS-LaribPolypDB  --out_pth Dataset/CVC/ETIS-LaribPolypDB/masks
# python scripts/segmentation_env.py --inp_pth Result/CVC_RES/CVC-300_V2  --out_pth Dataset/CVC/CVC-300/masks