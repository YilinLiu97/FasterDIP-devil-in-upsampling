import torch
import torch.nn as nn

import functools

class Unet_struc(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [ 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear')
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear')
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = torch.cat([f2, self.b(f2)], 1)
        f4 = torch.cat([f1, self.up2(f3)], 1)
        return self.up1(f4)

model = Unet_struc(3,3,256)
x = torch.rand(1,3,512,512)
out = model(x)
print(model)
print(out.shape)
