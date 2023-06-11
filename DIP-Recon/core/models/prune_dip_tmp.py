import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pylab as py
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils.common_utils import *

class Unet_struc(nn.Module):
    def __init__(self, save_path, out_size, input_nc, output_nc, ngf=64, upsample_mode='bilinear', adaptive_halting=False, ponder_lbda_p=0.5, ponder_epsilon=0.05, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.save_path = save_path
        
        use_bias = True
        
        model = {}
                
 #       if adaptive_halting:
  #         self.halting_unit1 = get_halting_prob(self.ngf)
           

        model['down8'] = [ 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            ]
        self.down8 = nn.Sequential(*model['down8'])


        model['down7'] = [ 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            ]
        self.down7 = nn.Sequential(*model['down7'])

        model['down6'] = [ 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            ]
        self.down6 = nn.Sequential(*model['down6'])

        model['down5'] = [ 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            ]
        self.down5 = nn.Sequential(*model['down5'])

        model['down4'] = [ 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            ]
        self.down4 = nn.Sequential(*model['down4'])

        model['down3'] = [ 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            ]
        self.down3 = nn.Sequential(*model['down3'])

        model['down2'] = [ 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            ]
        self.down1 = nn.Sequential(*model['down1'])

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

    def anytime_prediction(self, layer_out, target, layer_name, ground_truth):
        psd2d_layer, psd1d_layer = get_psd(layer_out, log=True)
        psd2d_target, psd1d_target = get_psd(target, log=True)

        # save the visualization       
        py.semilogy(psd1d_layer)
        py.semilogy(psd1d_target)
        py.xlabel("Spatial Frequency")
        py.ylabel("Power Spectrum")
        py.legend(['layer_output', 'target'])

        plt.savefig(f"{self.save_path}/{layer_name}_psd1D.png")
        plt.close()
        
        feat_out, ground_truth = layer_out[0].data.cpu().numpy(), ground_truth[0].data.cpu().numpy()
        scores = eval_general(ground_truth, feat_out)
       
        feat_out = feat_out.transpose(1,2,0)
        if ground_truth.shape[0] == 3:
           ground_truth = ground_truth.transpose(1,2,0)

        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(121)
        ax.imshow(ground_truth)
        ax.set_title(f"Ground Truth")
        ax.axis('off')

        ax = fig.add_subplot(122)
        ax.imshow(feat_out)
        ax.set_title('PSNR: %.4f  SSIM: %.4f'%(scores['psnr'], scores['ssim']))
        ax.axis('off')
        plt.savefig(f"{self.save_path}/{layer_name}_output.png")
        plt.close()

        return self.bce(psd1d_layer.detach(), psd1d_target.detach())

    def forward(self, input, target=None, ground_truth=None):
        f1 = self.down1(input)
        f1_out = self.up1(f1)
        if target is not None:
        #   f1_out = self.up1(f1)
           self.anytime_prediction(f1_out.detach(), target.detach(), 'down1', ground_truth)
        
        f2 = self.down2(f1)
        f2_out = self.up1(self.up2(f2))
        if target is not None:
       #    f2_out = self.up1(self.up2(f2))
           self.anytime_prediction(f2_out.detach(), target.detach(), 'down2', ground_truth)
        
        f3 = self.down3(f2)
        f3_out = self.up1(self.up2(self.up3(f3)))
        if target is not None:
       #    f3_out = self.up1(self.up2(self.up3(f3)))
           self.anytime_prediction(f3_out.detach(), target.detach(), 'down3', ground_truth)
        
        f4 = self.down4(f3)
        f4_out = self.up1(self.up2(self.up3(self.up4(f4))))
        if target is not None:
      #     f4_out = self.up1(self.up2(self.up3(self.up4(f4))))
           self.anytime_prediction(f4_out.detach(), target.detach(), 'down4', ground_truth)
        
        f5 = self.down5(f4)
        f5_out = self.up1(self.up2(self.up3(self.up4(self.up5(f5)))))
        if target is not None:
  #         f5_out = self.up1(self.up2(self.up3(self.up4(self.up5(f5)))))
           self.anytime_prediction(f5_out.detach(), target.detach(), 'down5', ground_truth)
        
        f6 = self.down6(f5)
        f6_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(f6))))))
        if target is not None:
#           f6_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(f6))))))
           self.anytime_prediction(f6_out.detach(), target.detach(), 'down6', ground_truth)

        f7 = self.down7(f6)
        f7_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(self.up7(f7)))))))
        if target is not None:
#           f7_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(self.up7(f7)))))))
           self.anytime_prediction(f7_out.detach(), target.detach(), 'down7', ground_truth)

        f8 = self.down8(f7)
       # f8_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(self.up7(self.up8(f8))))))))

#        if target is not None:
#           f8_out = self.up1(self.up2(self.up3(self.up4(self.up5(self.up6(self.up7(self.up8(f8))))))))
#           self.anytime_prediction(f8_out.detach(), target.detach(), 'down8', ground_truth)

        up8 = self.up8(f8)
        up7 = self.up7(up8)
        up6 = self.up7(up7)
        up5 = self.up6(up6)
        
        up4 = self.up5(up5)
        
        up3 = self.up4(up4)
        
        up2 = self.up3(up3)
        
        up1 = self.up2(up2)
        
        out = self.up1(up1)

        if ground_truth is None:
           return f1_out, f2_out, f3_out, f4_out, f5_out, f6_out, f7_out,  out        
        else:
           return out

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


