import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F

class ConvDropoutNormNonlin(nn.Module):

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class AdaptiveDepthStackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, max_num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):

        super(AdaptiveDepthStackedConvLayers, self).__init__()
        
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels


        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.max_num_convs = max_num_convs

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs
       
        self.block1 = basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs_first_conv,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs) 
        if max_num_convs >= 2:
            self.block2 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        if max_num_convs >= 3:
            self.block3 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        if max_num_convs >= 4:                
            self.block4 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        self._initialize_betas()

    def forward(self, x, tau):
        res_list = []
        out1 = self.block1(x)
        res_list.append(out1)
        if self.max_num_convs >= 2:
            out2 = self.block2(out1)
            res_list.append(out2)
        if self.max_num_convs >= 3:
            out3 = self.block3(out2)
            res_list.append(out3)
        if self.max_num_convs >= 4:
            out4 = self.block4(out3)
            res_list.append(out4)

        weights = F.gumbel_softmax(self.betas, tau=tau, dim=-1)
        assert len(weights) == len(res_list)
        out = sum(w*res for w, res in zip(weights, res_list))
        return out
    
    def _initialize_betas(self):
     
        assert self.max_num_convs in [2,3,4]
        betas = torch.zeros((self.max_num_convs))
        self.register_parameter('betas', nn.Parameter(betas))


x = torch.rand(1,3,512,512)
block = AdaptiveDepthStackedConvLayers(3, 3, 4)
out = block(x)
print(out.shape)
print(block)
