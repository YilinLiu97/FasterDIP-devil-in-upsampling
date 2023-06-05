import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pylab as py
import os
import sys
import inspect
import h5py

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.common_utils import *
from .common import *

def pad_to(t, padding, dim = -1, value = 0.):
    if dim > 0:
        dim = dim - t.ndim
    zeroes = -dim - 1
    return F.pad(t, (*((0, 0) * zeroes), *padding), value = value)

def safe_cumprod(t, eps = 1e-10, dim = -1):
    t = torch.clip(t, min = eps, max = 1.)
    return torch.exp(torch.cumsum(torch.log(t), dim = dim))

def exclusive_cumprod(t, dim = -1):
    cum_prod = safe_cumprod(t, dim = dim)
    return pad_to(cum_prod, (1, -1), value = 1., dim = dim)

def calc_geometric(l, dim = -1):
    return exclusive_cumprod(1 - l, dim = dim) * l

class RegularizationLoss(nn.Module):
    '''
        Computes the KL-divergence between the halting distribution generated
        by the network and a geometric distribution with parameter `lambda_p`.
        Parameters
        ----------
        lambda_p : float
            Parameter determining our prior geometric distribution.
        max_steps : int
            Maximum number of allowed pondering steps.
    '''

    def __init__(self, lambda_p: float, max_steps: int = 1_000, device=None):
        super().__init__()

        p_g = torch.zeros((max_steps,), device=device)
        not_halted = 1.

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.Tensor):
        '''
            Compute the loss.
            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, representing our
                halting distribution.
            Returns
            -------
            loss : torch.Tensor
                Scalar representing the regularization loss.
        '''
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        return self.kl_div(p.log(), p_g)

class Unet_struc(nn.Module):
    def __init__(self, save_path, out_size, input_nc, output_nc, ngf=64, upsample_mode='bilinear',
                 adaptive_halting=True, ponder_lbda_p=0.5, ponder_epsilon=0.005, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.save_path = save_path
        self.eps = ponder_epsilon
        self.ponder_lbda_p = ponder_lbda_p

#        self.geometric_prior = calc_geometric(torch.full((8,), ponder_lbda_p))
  #      self.reg_loss = RegularizationLoss(ponder_lbda_p, max_steps=8)

        use_bias = True

        model = {}

        if adaptive_halting:
           self.halting_unit1 = get_halting_prob(output_nc)
           self.halting_unit2 = get_halting_prob(output_nc)
           self.halting_unit3 = get_halting_prob(output_nc)
           self.halting_unit4 = get_halting_prob(output_nc)
           self.halting_unit5 = get_halting_prob(output_nc)
           self.halting_unit6 = get_halting_prob(output_nc)
           self.halting_unit7 = get_halting_prob(output_nc)
           self.halting_unit8 = get_halting_prob(output_nc)

        
        model['down8'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
        ]
        self.down8 = nn.Sequential(*model['down8'])

        model['exit8'] = [
            # up8
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up7
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up6
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up5
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up4
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up3
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up2
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up1
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.exit8 = nn.Sequential(*model['exit8'])


        model['down7'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
        ]
        self.down7 = nn.Sequential(*model['down7'])

        '''
        model['exit7'] = [
            # up7
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up6
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up5
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up4
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up3
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up2
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up1
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.exit7 = nn.Sequential(*model['exit7'])
        '''

        model['down6'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
        ]
        self.down6 = nn.Sequential(*model['down6'])

        '''
        model['exit6'] = [
            # up6
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up5
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up4
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up3
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up2
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up1
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.exit6 = nn.Sequential(*model['exit6'])
        '''

        model['down5'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
        ]
        self.down5 = nn.Sequential(*model['down5'])

        '''
        model['exit5'] = [
            # up5
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up4
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up3
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up2
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up1
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.exit5 = nn.Sequential(*model['exit5'])
        '''
        model['down4'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
        ]
        self.down4 = nn.Sequential(*model['down4'])

        '''
        model['exit4'] = [
            # up4
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up3
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up2
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up1
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.exit4 = nn.Sequential(*model['exit4'])
        '''
 
        model['down3'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
        ]
        self.down3 = nn.Sequential(*model['down3'])

        '''
        model['exit3'] = [
            # up3
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up2
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up1
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.exit3 = nn.Sequential(*model['exit3'])
        '''

        model['down2'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
        ]
        self.down2 = nn.Sequential(*model['down2'])

        model['exit2'] = [
            # up2
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            # up1
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.exit2 = nn.Sequential(*model['exit2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
        ]
        self.down1 = nn.Sequential(*model['down1'])

        '''
        model['exit1'] = [
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.exit1 = nn.Sequential(*model['exit1'])

        '''
        model['up8'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
        ]
        self.up8 = nn.Sequential(*model['up8'])

        model['up7'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
        ]
        self.up7 = nn.Sequential(*model['up7'])

        model['up6'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
        ]
        self.up6 = nn.Sequential(*model['up6'])
        
   
        model['up5'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
        ]
        self.up5 = nn.Sequential(*model['up5'])
        

        model['up4'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
        ]
        self.up4 = nn.Sequential(*model['up4'])

        model['up3'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
        ]
        self.up3 = nn.Sequential(*model['up3'])

        model['up2'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
        ]
        self.up2 = nn.Sequential(*model['up2'])

        model['up1'] = [
            nn.Upsample(size=out_size, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
        ]
        self.up1 = nn.Sequential(*model['up1'])

        self.bce = nn.BCELoss()

    def anytime_prediction(self, layer_out, target, layer_name, ground_truth, save_path):
        psd2d_layer, psd1d_layer = get_psd(layer_out, log=True)
        psd2d_target, psd1d_target = get_psd(target, log=True)

        print(">>>>>>>>>>>>> Save path: ", save_path)
        # save the visualization
        py.semilogy(psd1d_layer)
        py.semilogy(psd1d_target)
        py.xlabel("Spatial Frequency")
        py.ylabel("Power Spectrum")
        py.legend(['layer_output', 'target'])

        plt.savefig(f"{save_path}/{layer_name}_psd1D.png")
        plt.close()

        feat_out, ground_truth = layer_out[0].data.cpu().numpy(), ground_truth[0].data.cpu().numpy()
        scores = eval_general(ground_truth, feat_out)

        feat_out = feat_out.transpose(1, 2, 0)
        if ground_truth.shape[0] == 3:
            ground_truth = ground_truth.transpose(1, 2, 0)

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121)
        ax.imshow(ground_truth)
        ax.set_title(f"Ground Truth")
        ax.axis('off')

        ax = fig.add_subplot(122)
        ax.imshow(feat_out)
        ax.set_title('PSNR: %.4f  SSIM: %.4f' % (scores['psnr'], scores['ssim']))
        ax.axis('off')
        plt.savefig(f"{save_path}/{layer_name}_output.png")
        plt.close()

        hf = h5py.File(f"{save_path}/{layer_name}_psd_output.h5", 'w')
        hf.create_dataset('psd1d_output', data=psd1d_layer)
        hf.create_dataset('psd1d_target', data=psd1d_target)
        hf.create_dataset('psd2d_output', data=psd2d_layer)
        hf.create_dataset('psd2d_target', data=psd2d_target)
        hf.close()

        return self.bce(torch.from_numpy(psd1d_layer).detach(), torch.from_numpy(psd1d_target).detach())

    def forward(self, input, tau=1, target=None, ground_truth=None, training=True, save_path=None):
        cum, ponder_cost, R = 0, 0, 1
       # halting_logits = []

       # print(">>>>>>>>>>>>> tau: ", tau)
        f1 = self.down1(input)
       # f1_out = self.exit1(f1)
       # halting_logits.append(self.halting_unit1(f1_out))

        f2 = self.down2(f1)
       # halting_logits.append(self.halting_unit2(f2_out))
        
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        f5 = self.down5(f4)
        f6 = self.down6(f5)
        f7 = self.down7(f6)
        f8 = self.down8(f7)
        up8 = self.up8(f8)
        up7 = self.up7(up8)
        up6 = self.up6(up7)
        up5 = self.up5(up6)
        up4 = self.up4(up5)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        out = self.up1(up2)
        
       # f3_out = self.exit3(f3)
       # halting_logits.append(self.halting_unit3(f3_out))
        '''
        f4 = self.down4(f3)
       # f4_out = self.exit4(f4)
       # halting_logits.append(self.halting_unit4(f4_out))

        f5 = self.down5(f4)
       # f5_out = self.exit5(f5)
       # halting_logits.append(self.halting_unit5(f5_out))

        f6 = self.down6(f5)
       # f6_out = self.exit6(f6)
       # halting_logits.append(self.halting_unit6(f6_out))

        
        f7 = self.down7(f6)
       # f7_out = self.exit7(f7)
       # halting_logits.append(self.halting_unit7(f7_out))

        f8 = self.down8(f7)
        f8_out = self.exit8(f8) 
        
       # halting_logits.append(self.halting_unit8(f8_out))

       # halting_logits = torch.stack(halting_logits, dim=1) #[0]
 
       # collect = [f1_out, f2_out, f3_out, f4_out, f5_out, f6_out, f7_out, f8_out]
        '''
        if training:
           #hp = cal_geometric(halting_logits.sigmoid(), dim = 1) #F.gumbel_softmax(hp, tau=tau, dim=-1, hard=True) 
           return out #f2_out, f8_out #, f3_out, f4_out , f5_out , f6_out , f7_out, f8_out
        else:
           f8_out = out.detach()
          # self.anytime_prediction(f1_out, target.detach(), 'down1', ground_truth, save_path)
          # self.anytime_prediction(f2_out.detach(), target.detach(), 'down2', ground_truth, save_path)
          # self.anytime_prediction(f3_out.detach(), target.detach(), 'down3', ground_truth, save_path)
          # self.anytime_prediction(f4_out.detach(), target.detach(), 'down4', ground_truth, save_path)
          # self.anytime_prediction(f5_out.detach(), target.detach(), 'down5', ground_truth, save_path)
          # self.anytime_prediction(f6_out.detach(), target.detach(), 'down6', ground_truth, save_path)
          # self.anytime_prediction(f7_out.detach(), target.detach(), 'down7', ground_truth, save_path)
           self.anytime_prediction(f8_out.detach(), target.detach(), 'down8', ground_truth, save_path)

           return out
        
    def forward2(self, input, tau=1, target=None, ground_truth=None, training=True):
        cum, ponder_cost, R = 0, 0, 1

        f1 = self.down1(input)
        f1_out = self.up1(f1)
        h1 = self.halting_unit1(f1_out)
        print('h1: ', h1)
        cum += h1
        ponder_cost += 1
        
        if cum >= 1 - self.eps:
           ponder_cost += R
           return f1_out, 'down1', ponder_cost
        else:
            R -= h1

        f2 = self.down2(f1)
        f2_out = self.up1(self.up2(f2))
        h2 = self.halting_unit2(f2_out)
        print('h2: ', h2)
        cum += h2
        ponder_cost += 1
        
        if cum >= 1 - self.eps:
           ponder_cost += R
           return f2_out, 'down2', ponder_cost
        else:
            R -= h2

        f3 = self.down3(f2)
        f3_out = self.up1(self.up2(self.up3(f3)))
        print('h3: ', h3)
        cum += h3
        ponder_cost += 1
        
        if cum >= 1 - self.eps:
            ponder_cost += R
            return f3_out, 'down3', ponder_cost
        else:
            R -= h3

        f4 = self.down4(f3)
        f4_out = self.up1(self.up2(self.up3(self.up4(f4))))
        h4 = self.halting_unit4(f4_out)
        cum += h4
        ponder_cost += 1
        if cum >= 1 - self.eps:
            ponder_cost += R
            return f4_out, 'down4', ponder_cost
        else:
            R -= h4

        f5 = self.down5(f4)
        f5_out = self.up1(self.up2(self.up3(self.up4(self.up5(f5)))))
        h5 = self.halting_unit5(f5_out)
        cum += h5
        ponder_cost += 1
        if cum >= 1 - self.eps:
            ponder_cost += R
            return f5_out, 'down5', ponder_cost
        else:
            R -= h5

        f6 = self.down6(f5)
        f6_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(f6))))))
        h6 = self.halting_unit6(f6_out)
        cum += h6
        ponder_cost += 1
        if cum >= 1 - self.eps:
            ponder_cost += R
            return f6_out, 'down6', ponder_cost
        else:
            R -= h6

        f7 = self.down7(f6)
        f7_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(self.up7(f7)))))))
        h7 = self.halting_unit7(f7_out)
        cum += h7
        ponder_cost += 1
        if cum >= 1 - self.eps:
            ponder_cost += R
            return f7_out, 'down7', ponder_cost
        else:
            R -= h7

        f8 = self.down8(f7)
        f8_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(self.up7(self.up8(f8))))))))
        h8 = self.halting_unit8(f8_out)
        ponder_cost += 1

        return f8_out, 'final', ponder_cost


def network_info(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(f"Total number of params: {num_params}")


'''
target = torch.rand(1,3,256,384)
x = torch.rand(1,128,256,384)
model = Unet_struc('../saves/denoising/DIP_2_scaled_2levels_0skips_128chns_AllImages/snail', x.shape[-2:], 128,3,128)
out = model(x, target)
print(model)
print(out.shape)

network_info(model)
'''




