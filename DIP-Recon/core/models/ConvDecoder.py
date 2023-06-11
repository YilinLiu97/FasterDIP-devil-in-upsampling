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
        
        # Added by Jiang  Print some information 
        print('===============================ConvDecoder ==================================================')
        print(f'num_layers:{num_layers} num_channels:{num_channels} out_size:{out_size} in_size:{in_size}\n'
        f'act_func:{act_func} upsample_mode:{upsample_mode} need_dropout:{need_dropout} need_sigmoid:{need_sigmoid} norm_func:{norm_func}')
        print('===============================ConvDecoder ==================================================')
        
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
        
         # Rectangle window
        w1 = torch.tensor([0.5, 0.5])
        # Triangle window
        w2 = torch.tensor([0.25, 0.5, 0.25])
        # -20dB LPF 7 taps
        w3 = torch.tensor([-0.1070,    0.0000,    0.3389,    0.5360,    0.3389,    0.0000,   -0.1070])
        
        w41 = torch.tensor([ 0.0027,    0.0140,    0.0113,   -0.0217,   -0.0556,   -0.0218,    0.1123,    0.2801,    0.3572,    0.2801,    0.1123 ,  -0.0218,   -0.0556,   -0.0217,    0.0113,    0.0140,    0.0027])

        w4 = torch.tensor([0.002745,0.014011,0.011312,-0.021707,-0.055602,-0.021802,0.112306,0.280146,0.357183,0.280146,0.112306,-0.021802,-0.055602,-0.021707,0.011312,0.014011,0.002745])
        w14 = torch.tensor([-0.0054,-0.0518,0.2554,0.6036,0.2554,-0.0518, -0.0054])
        w15 = torch.tensor([0.0105,  -0.0263,   -0.0518,    0.2763,    0.5826,    0.2763,   -0.0518, -0.0263, 0.0105])
        w5 = torch.tensor([-0.001921,-0.004291,0.009947,0.021970,-0.022680,-0.073998,0.034907,0.306100,0.459933,0.306100,0.034907,-0.073998,-0.022680,0.021970,0.009947,-0.004291,-0.001921]) 
        w6 = torch.tensor([0.000015,0.000541,0.003707,0.014130,0.037396,0.075367,0.121291,0.159962,0.175182,0.159962,0.121291,0.075367,0.037396,0.014130,0.003707,0.000541,0.000015])
        
        w_list = [w3, w3, w3] 

        for i in range(num_layers - 1):

#            self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))

            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2,
                             bias=True)
            self.net.add(conv)
            # Updated by Jiang
            #self.net.add(zero_insertion_lowpass_conv(num_channels, pad_size=(kernel_size - 1) // 2, outsize=hidden_size[i][0]))
            if upsample_mode == 'nearest' or upsample_mode == 'bilinear' or upsample_mode == 'bicubic':
                self.net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            else:
                self.net.add(InsertZeros(2, 2, gain=1.0))
                if upsample_mode == 'LPF1':
                    self.net.add(lowpass_conv3(num_channels, w1, pad_mode = 'reflect', gain=4.0))
                    
                elif upsample_mode == 'LPF2':
                    self.net.add(lowpass_conv3(num_channels, w2, pad_mode = 'reflect', gain=4.0))
                
                elif upsample_mode == 'LPF3':
                    self.net.add(lowpass_conv3(num_channels, w3, pad_mode = 'reflect', gain=4.0))
                
                elif upsample_mode == 'LPF4':
                    self.net.add(lowpass_conv3(num_channels, w4, pad_mode = 'reflect', gain=4.0))
                
                elif upsample_mode == 'LPF41':
                    self.net.add(lowpass_conv3(num_channels, w41, pad_mode = 'reflect', gain=4.0))

                elif upsample_mode == 'LPF14':
                    self.net.add(lowpass_conv3(num_channels, w14, pad_mode = 'reflect', gain=4.0))

                elif upsample_mode == 'LPF15':
                    self.net.add(lowpass_conv3(num_channels, w15, pad_mode = 'reflect', gain=4.0))

                elif upsample_mode == 'LPF5':
                    self.net.add(lowpass_conv3(num_channels, w5, pad_mode = 'reflect', gain=4.0))
                
                elif upsample_mode == 'LPF6':
                    self.net.add(lowpass_conv3(num_channels, w6, pad_mode = 'reflect', gain=4.0))
                
                else:
                    print(f'Upsample Mode:{upsample_mode} not supported! Change to bilinear interpolation')
                    #self.net.add(nn.Upsample(size=hidden_size[i], mode='bilinear'))
                    self.net.add(lowpass_conv3(num_channels, w2, pad_mode = 'reflect', gain=4.0))
                
                    
            
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

