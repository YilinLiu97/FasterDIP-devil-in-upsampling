import sys

import torch
import torch.nn as nn
import numpy as np
from .common import *

sys.path.append('../')
from pruning.DAM import * 

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
torch.nn.Module.add = add_module


class ConvDecoder(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size, act_func='ReLU', upsample_mode='nearest', norm_func='bn', need_dropout=False,
                 need_sigmoid=False):
        super(ConvDecoder, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1] * (num_layers - 1)

        norm_layer = nn.BatchNorm2d if norm_func == 'bn' else nn.InstanceNorm2d

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]
        print(hidden_size)
        ### hidden layers
        self.net = nn.Sequential()

        for i in range(num_layers - 1):
            if upsample_mode == 'transposed':
               self.net.add(nn.ConvTranspose2d(num_channels, num_channels, kernel_size=2, stride=2))
            elif upsample_mode == 'None':
               pass
            else:
               self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            
            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
#            self.net.add(learnable_zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2, outsize=hidden_size[i][0]))
            self.net.add(act(act_func))
            self.net.add(norm_layer(num_channels)) #nn.BatchNorm2d(num_channels, affine=True))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
       # self.net.add(zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2))
        self.net.add(act(act_func))
        self.net.add(norm_layer(num_channels))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
         
        if need_sigmoid:
          self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out
        
class ConvDecoder_pip(nn.Module):
    def __init__(self, input_dim, num_layers, num_channels, num_output_channels, out_size, in_size, kernel_size=1, act_func='ReLU', upsample_mode='nearest', norm_func='bn', need_dropout=False,
                 need_sigmoid=False):
        super(ConvDecoder_pip, self).__init__()

        ### parameter setup
        strides = [1] * (num_layers - 1)

        norm_layer = nn.BatchNorm2d if norm_func == 'bn' else nn.InstanceNorm2d

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]
        print(hidden_size)
        ### hidden layers
        self.net = nn.Sequential()

        for i in range(num_layers - 1):

#            self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            if i==0:
               conv = nn.Conv2d(input_dim, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            else:
               conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
#            self.net.add(learnable_zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2, outsize=hidden_size[i][0]))
            self.net.add(act(act_func))
            self.net.add(norm_layer(num_channels)) #nn.BatchNorm2d(num_channels, affine=True))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
       # self.net.add(zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2))
        self.net.add(act(act_func))
        self.net.add(norm_layer(num_channels))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
         
        if need_sigmoid:
          self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out

class ConvDecoder_improved(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size, act_func='ReLU', upsample_mode='nearest', norm_func='bn', need_dropout=False,
                 need_sigmoid=False, num_ups=4):
        super(ConvDecoder_improved, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1] * (num_layers - 1)

        norm_layer = nn.BatchNorm2d if norm_func == 'bn' else nn.InstanceNorm2d

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]

        ### hidden layers
        self.net = nn.Sequential()

       # for _ in range(4):
        #    self.net.add(nn.Conv2d(num_channels, num_channels, kernel_size, 1, padding=(kernel_size - 1) // 2, bias=True))
         #   self.net.add(act(act_func))
          #  self.net.add(norm_layer(num_channels))

        for i in range(num_layers - 1):
            if i == num_layers - 3:
##            if i==num_layers-3:
#               self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
               self.net.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            if i == num_layers - 2:
               self.net.add(nn.Upsample(size=out_size, mode=upsample_mode))
            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
            self.net.add(act(act_func))
            self.net.add(norm_layer(num_channels)) #nn.BatchNorm2d(num_channels, affine=True))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
        self.net.add(norm_layer(num_channels))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
        if need_sigmoid:
            self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out

"""
Prunable with DAM
"Learning Compact Representations of Neural Networks using DiscriminAtive Masking (DAM)"
"""
class ConvDecoder_DAM(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size, act_func='ReLU', upsample_mode='nearest', norm_func='bn', need_dropout=False,
                 need_sigmoid=False):
        super(ConvDecoder_DAM, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1] * (num_layers - 1)

        norm_layer = nn.BatchNorm2d if norm_func == 'bn' else nn.InstanceNorm2d

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]

        ### hidden layers
        self.net = nn.Sequential()
        for i in range(num_layers - 1):

            self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))

            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
            self.net.add(act(act_func))
            self.net.add(norm_layer(num_channels))
            self.net.add(DAM_2d(num_channels))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
        self.net.add(norm_layer(num_channels))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
        if need_sigmoid:
            self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out

"""
Y-shape with two different branches leading to different outputs.
Weights are forced to be orthogonal by min | WW' - I |
"""
class BiConvDecoder(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size, act_func='ReLU', upsample_mode='nearest', need_dropout=False,
                 need_sigmoid=False):
        super(BiConvDecoder, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1] * (num_layers - 1)

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                    1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]

        ### hidden layers
        self.net = nn.Sequential()
        for i in range(num_layers - 1):

            self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))

            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
            self.net.add(act(act_func))
            self.net.add(nn.BatchNorm2d(num_channels, affine=True))
            if need_dropout:
                self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(
            nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True))
        self.net.add(nn.BatchNorm2d(num_channels, affine=True))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
        if need_sigmoid:
            self.net.add(nn.Sigmoid())

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out





