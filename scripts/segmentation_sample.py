import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from ssl import OP_NO_TLSv1
import nibabel as nib
import sys
import random
sys.path.append(".")
sys.path.append("..")
import numpy as np
import time
import torch as th
from PIL import Image
import matplotlib.pyplot as plt
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import *
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.myloader import MyDataset
from guided_diffusion.BYYAloader import BYYADataset
from guided_diffusion.kvasir_loader import KvasirDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple,Visual_tensor
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
from torchmetrics import Dice, JaccardIndex
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
from torch.autograd import Function
import medpy.metric.binary as mc

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

def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

def calculate_metrics(predict_image, gt_image):
    # 将图像转换为二进制数组
    predict_image = np.array(predict_image, dtype=bool)
    gt_image = np.array(gt_image, dtype=bool)

    # 计算True Positive（TP）
    tp = np.sum(np.logical_and(predict_image, gt_image))

    # 计算True Negative（TN）
    tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(gt_image)))

    # 计算False Positive（FP）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(gt_image)))

    # 计算False Negative（FN）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), gt_image))

    # 计算IOU（Intersection over Union）
    iou = tp / (tp + fn + fp + 1e-7)
    # iou =( tp/(tp + fp + fn) + tn/(tn + fn + fp))/2

    # 计算Dice Coefficient（Dice系数）
    dice_coefficient = 2 * tp / (2 * tp + fn + fp + 1e-7)

    # 计算Accuracy（准确率）
    accuracy = (tp+tn) / (tp + fp + tn + fn + 1e-7)

    # 计算precision（精确率）
    precision = tp / (tp + fp + 1e-7)

    # 计算recall（召回率）
    recall = tp / (tp + fn + 1e-7)

    # 计算Sensitivity（敏感度）
    sensitivity = tp / (tp + fn + 1e-7)

    # 计算F1-score
    f1 = 2*(precision*recall)/(precision+recall + 1e-7)

    # 计算Specificity（特异度）
    specificity = tn / (tn + fp + 1e-7)
    hd = mc.hd(np.array(predict_image, dtype=float),np.array(gt_image, dtype=float))
    mpa = (tp+tn)/(fn+tp+fp+tn)

    return {
        "mIOU":iou,
        "dice_coefficient":dice_coefficient,
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "sensitivity":sensitivity,
        "f1":f1,
        "specificity":specificity,
        'hd':hd,
        'mpa':mpa
    }

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), transforms.Normalize([0.706768,0.53929335,0.49650374],[0.15592906,0.17086771,0.19545093])]
        transform_test_img = transforms.Compose(tran_list)
        transform_test_mask = transforms.Compose(tran_list[:-1])
        ds = ISICDataset(args.data_dir, transform_test_img,transform_test_mask,mode='Test')
        args.in_ch = 4
    elif args.data_name == 'KVASIR':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), transforms.Normalize([0.55715865,0.32168046,0.23578914],[0.31722644,0.21997158,0.18551587])]
        transform_train_img = transforms.Compose(tran_list[:-1])
        transform_train_mask = transforms.Compose(tran_list[:-1])
        ds = KvasirDataset(args.data_dir, transform_train_img,transform_train_mask)
        args.in_ch = 4

    if args.data_name == 'MY':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)
        ds = MyDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4
    
    elif args.data_name == 'BYYA':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), transforms.Normalize([0.70844054,0.5821276,0.53606266],[0.15572749,0.16400227,0.17911644])]
        transform_train_img = transforms.Compose(tran_list)
        transform_train_mask = transforms.Compose(tran_list[:-1])
        ds = BYYADataset(args.data_dir, transform_train_img,transform_train_mask,mode='Testing')
        args.in_ch = 4

    elif args.data_name == 'BRATS':
        # tran_list = [myNormalize(mean=[ 318.7704, 1507.5906,  846.4935,  268.6659],std=[ 2541.7253, 45207.7148, 16756.3789,  3519.8733]),myResize(args.image_size,args.image_size)]
        tran_list = [myNormalize(mean=[246.0692, 208.3315, 338.0198, 165.9171],std=[716.8119, 635.6193, 941.2881, 642.6382]),myResize(args.image_size,args.image_size)]
        transform_test = transforms.Compose(tran_list)
        ds = BRATSDataset_5H(args.data_dir,transform_test)
        args.in_ch = 5
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion, distribution = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    # model.to(dist_util.dev())
    model.to(torch.device("cuda:1"))
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # 初始化指标累积器
    total_metrics = {
        "mIOU": 0.0,
        "dice_coefficient": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "sensitivity": 0.0,
        "f1": 0.0,
        "specificity": 0.0,
        'hd': 0.0,
        'mpa': 0.0
    }
    num_samples = 0
    
    for _ in range(len(data)):
        b, m, path = next(data)  #should return an image from the dataloader "data", 返回的是image, mask, name
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'KVASIR':
            slice_ID = path[0].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            # slice_ID=path[0].split("_")[-1] + "_" + path[0].split("slice")[-1].split('.nii')[0]
            slice_ID=path[0].split("_")[-1]
        
        elif args.data_name == 'MY':
            slice_ID = path[0].split(".")[0]
        
        elif args.data_name == 'BYYA':
            slice_ID = path[0].split('/')[-1].split('.')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, args.in_ch, args.image_size, args.image_size), 
                img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            logger.log('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            # co = th.tensor(cal_out)
            # if args.version == 'v1':
            #     enslist.append(sample[:,-1,:,:])
            # else:
            #     enslist.append(co)
            enslist.append(sample[:,-1,:,:])
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0) #torch.size为(256,256)
        
        # ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)
        ensres_np = ensres.detach().cpu().numpy()
        m_np = m.squeeze(0).squeeze(0).detach().cpu().numpy()

        # 将预测结果和真实标签转换为二值图像
        threshold = 0.5 
        predict_binary = (ensres_np > threshold).astype(np.uint8)
        gt_binary = (m_np > threshold).astype(np.uint8)

        # 计算指标
        metrics = calculate_metrics(predict_binary, gt_binary)

        # 累积指标
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_samples += 1

        # 可选：每个样本打印指标
        logger.log(f"Sample {num_samples} metrics: {metrics}")

    # 循环结束后，计算平均指标
    average_metrics = {key: total_metrics[key] / num_samples for key in total_metrics}

    logger.log("Average Metrics over all samples:")
    logger.log(average_metrics)

def create_argparser():
    defaults = dict(
        data_name = 'BYYA',
        data_dir="/media/kemove/sata/TIANXIAO/tianxiao/DFMIS/Dataset/test",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/',
        multi_gpu = "0,1", #"0,1,2"
        debug = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
