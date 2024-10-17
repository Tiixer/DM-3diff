'''
    Wavelets Trasform + Residual dense block + Relu Block
'''
import torch
import torch.nn as nn
from .RDB import RDB
from pytorch_wavelets import DWTForward, DWTInverse   # (or import DWT, IDWT)

class WRB(nn.Module):
    def __init__(self, in_channels, growth_rate=32, RDB_num_layers=3, wave='coif3', lrl=True):
        super(WRB, self).__init__()

        n_feats = in_channels*6
        growth_rate = growth_rate
        num_layers = RDB_num_layers
        lrl = lrl
        self.calculate_weight = nn.Parameter(torch.rand(1,dtype=torch.float32))
        if lrl:
            self.RDB = RDB(n_feats, growth_rate, num_layers, lrl=lrl)
            self.conv2 = nn.Conv2d(n_feats, in_channels*3, kernel_size=1)
        else:
            self.RDB = RDB(n_feats, growth_rate, num_layers, lrl=lrl)
            self.conv2 = nn.Conv2d(n_feats+num_layers*growth_rate, in_channels*3, kernel_size=1)

        # J为分解的层次数,wave表示使用的变换方法
        self.WTF = DWTForward(J=1, mode='zero', wave=wave)  # Accepts all wave types available to PyWavelets
        self.WTI = DWTInverse(mode='zero', wave=wave)

    def forward(self, x, hs=None):
        batch_size, _, h, w = x.shape

        if h % 2 == 1 and w % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 0:-1, 0:-1]
        elif h % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 0:-1, 1:-1]
        elif w % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 1:-1, 0:-1]

        yl_x, yh_x = self.WTF(x)
        yl_hs, yh_hs = self.WTF(hs)

        yh_x = yh_x[0] 
        fh_x, fw_x = yh_x.shape[-2], yh_x.shape[-1]
        yh_x = yh_x.view(batch_size, -1, fh_x, fw_x)

        yh_hs = yh_hs[0] 
        fh_hs, fw_hs = yh_hs.shape[-2], yh_hs.shape[-1]
        yh_hs = yh_hs.view(batch_size, -1, fh_hs, fw_hs)

        out = torch.cat((yh_x , yh_hs * self.calculate_weight), 1)

        out2 = self.RDB(out)
        out3 = self.conv2(out2)
        yl = yl_x + yl_hs
        yh = out3.view(batch_size, -1, 3, fh_x, fw_x)
        yh = [yh, ]
        out = self.WTI((yl, yh))

        if h % 2 == 1 and w % 2 == 1:
            out = out[:, :, 1:, 1:]
        elif h % 2 == 1:
            out = out[:, :, 1:, :]
        elif w % 2 == 1:
            out = out[:, :, :, 1:]
        return out