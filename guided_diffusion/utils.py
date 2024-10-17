import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from visdom import Visdom
import matplotlib.pyplot as plt
import seaborn as sns
softmax_helper = lambda x: F.softmax(x, 1)
sigmoid_helper = lambda x: F.sigmoid(x)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def staple(a):
    # a: n,c,h,w detach tensor
    # mvres = mv(a)
    # gap = 0.4
    # if gap > 0.001:
    #     for i, s in enumerate(a):
    #         r = s * mvres
    #         res = r if i == 0 else torch.cat((res,r),0)
    #     nres = mv(res)
    #     gap = torch.mean(torch.abs(mvres - nres))
    #     mvres = nres
    #     a = res
    for i,s in enumerate(a):
        res = s if i == 0 else s + res
    return std(norm(res))

def allone(disc,cup):
    disc = np.array(disc) / 255
    cup = np.array(cup) / 255
    res = np.clip(disc * 0.5 + cup,0,1) * 255
    res = 255 - res
    res = Image.fromarray(np.uint8(res))
    return res

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim=True) / b

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

def export(tar, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tar.size(1)
    if c == 3:
        vutils.save_image(tar, fp = img_path)
    else:
        s = th.tensor(tar)[:,-1,:,:].unsqueeze(1)
        s = th.cat((s,s,s),1)
        vutils.save_image(s, fp = img_path)

def norm(t):
    m, s, v = torch.mean(t), torch.std(t), torch.var(t)
    return (t - m) / s


def std(t):
    max,min = torch.max(t), torch.min(t)
    return (t-min)/(max-min)

def Visual_tensor(input_tensor,indexlist,transposelist,sublist,batchsize,mode='tensor'):
    vis = Visdom(port=8850)
    if mode == 'image':
        temp_input = input_tensor.cpu().numpy()  #B*C*H*W
        vis.images(temp_input,nrow=4)
    else:
        temp_input = input_tensor.cpu().numpy()
        plt.figure(figsize=(3,3))
        for i in range(batchsize):
            for j in range(len(indexlist)):
                plt.subplot(sublist[0],sublist[1],j+1)
                plt.imshow(temp_input[i][j],cmap='gray')
                plt.colorbar()
            vis.matplot(plt)

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg
