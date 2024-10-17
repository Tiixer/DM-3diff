import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from .unet import Generic_UNet
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels_dist, enc_num_filters, dec_num_filters, latent_dim):
        super(AxisAlignedConvGaussian, self).__init__()
        self.encoder = Generic_UNet(input_channels_dist, 32,1,5)
        self.latent_dim = latent_dim
        self.encoder_conv_layers = [nn.Conv2d(x, 2 * self.latent_dim, (1,1), stride=1).to(device) for x in enc_num_filters]
        self.decoder_conv_layers = [nn.Conv2d(x, 2 * self.latent_dim, (1,1), stride=1).to(device) for x in dec_num_filters]

    def forward(self, input):
        
        if type(input) == list:
            decoder_emb_list = []
            input = [torch.mean(x, dim=2, keepdim=True) for x in input]
            input = [torch.mean(x, dim=3, keepdim=True) for x in input]
            for i, item in enumerate(input):
                decoder_emb_list.append(self.decoder_conv_layers[i](item))
            decoder_emb_list = [torch.squeeze(x, dim=2) for x in decoder_emb_list]
            decoder_emb_list = [torch.squeeze(x, dim=2) for x in decoder_emb_list]
            decoder_mu_log_sigma = [(x[:,:self.latent_dim],x[:,self.latent_dim:]) for x in decoder_emb_list]
            decoder_dist = [Independent(Normal(loc=x[0], scale=torch.exp(x[1]) + 1e-6), 1) for x in decoder_mu_log_sigma]

            # decoder_dist = [Independent(Normal(loc=x[0], scale=torch.exp(x[1])),1) for x in decoder_mu_log_sigma]
            return decoder_dist
        else:
            emb, seg_outputs, hs_list = self.encoder(input,hs=None)
            input = hs_list[:-1]
            encoder_emb_list = []
            input = [torch.mean(x, dim=2, keepdim=True) for x in input]
            input = [torch.mean(x, dim=3, keepdim=True) for x in input]
            for i, item in enumerate(input):
                encoder_emb_list.append(self.encoder_conv_layers[i](item))
            encoder_emb_list = [torch.squeeze(x, dim=2) for x in encoder_emb_list]
            encoder_emb_list = [torch.squeeze(x, dim=2) for x in encoder_emb_list]
            encoder_mu_log_sigma = [(x[:,:self.latent_dim],x[:,self.latent_dim:]) for x in encoder_emb_list]
            encoder_dist = [Independent(Normal(loc=x[0], scale=torch.exp(x[1])),1) for x in encoder_mu_log_sigma]
            return encoder_dist
