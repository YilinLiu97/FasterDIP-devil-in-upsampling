import torch
import torch.nn as nn

class Unet_struc(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, upsample_mode='bilinear', norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        use_bias = True
        
        model = {}

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

        model['up2'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            norm_layer(self.ngf),
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1),
            nn.Sigmoid()
            ]
        self.up1 = nn.Sequential(*model['up1'])
        

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        up2 = torch.cat((f2, self.up2(f2)), 1)
        
        return torch.cat((f1,self.up1(up2)),1) 

def network_info(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(f"Total number of params: {num_params}")

model = Unet_struc(128,3,128)
x = torch.rand(1,128,256,384)
out = model(x)
print(model)
print(out.shape)

network_info(model)
print(model.up1.named_children())

