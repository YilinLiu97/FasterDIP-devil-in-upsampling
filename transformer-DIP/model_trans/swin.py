from model_trans.transformer import SwinTransformerSys
from torch import nn
import torch


class DIPModel(nn.Module):
    def __init__(self, img_size, input_chns=32, chns=3, patch_size=4, num_heads_down=[4, 4, 4],
                 num_heads_up=[1, 1, 1], need_sigmoid=True, upsample_mode='bilinear'):
        super().__init__()

        self.model = SwinTransformerSys(img_size=img_size, patch_size=patch_size, in_chans=input_chns)
        # self.model = skip(input_channels, output_channels, num_channels_down=[128] * 5, num_channels_up=[128] * 5,
        #                   num_channels_skip=[4] * 5)
        # print(self.model)
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        # self.model.apply(self.init_weights)

        self.l1 = nn.MSELoss()
        self.out_avg = None
        self.out = None

    def update_net(self, x, y, reg_noise_std=1./50, exp_weight=0.99, epoch=0):

        if reg_noise_std > 0:
            noise = x.detach().clone()
            net_input = x + (noise.normal_() * reg_noise_std)
        else:
            net_input = x

        self.out = self.model(net_input)

        # Smoothing
        if self.out_avg is None or epoch <= 100:
            self.out_avg = self.out.detach().cpu()
        else:
            self.out_avg = self.out_avg * exp_weight + self.out.detach().cpu() * (1 - exp_weight)

        total_loss = self.l1(self.out, y)
        total_loss.backward()

        return total_loss
