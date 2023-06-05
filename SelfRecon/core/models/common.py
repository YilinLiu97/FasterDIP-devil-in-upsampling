import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
import math
from .downsampler import Downsampler
from torch.nn import Parameter
from deepsplines.ds_modules import dsnn
import ptwt
import pywt

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class get_halting_prob(nn.Module):
    def __init__(self, dim):
        super(get_halting_prob, self).__init__()
        self.halting_unit = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        halting_prob = self.halting_unit(x)
        return halting_prob.flatten()

class BinarizerSTEStatic(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor. Backward is STE"""

    @staticmethod
    def forward(ctx, threshold, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        gradInput = gradOutput.clone()
        #gradInput.zero_()

        return None,gradInput

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun='LeakyReLU', chns=128):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        if act_fun == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'GELU':
            return nn.GELU()
        elif act_fun == 'DeepSpline':
            opt_params = {
            'size': 51,
            'range_': 4,
            'init': 'leaky_relu',
            'save_memory': False
            }
            return dsnn.DeepBSpline('conv', chns, **opt_params)
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.bias = Parameter(torch.Tensor(num_features))
        self.bias.data.zero_()

    def forward(self, inp):
        size = list(inp.size())
        beta = self.bias.view(1, self.num_features, 1, 1)
        avg = torch.mean(inp.view(size[0], self.num_features, -1), dim=2)

        output = inp - avg.view(size[0], size[1], 1, 1)
        output = output + beta

        return output

def bn(num_features):
#    return MeanOnlyBatchNorm(num_features) #nn.BatchNorm2d(num_features)
    return nn.BatchNorm2d(num_features)

# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), 0.5)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class InfinityNorm(nn.Module):
    def __init__(self, module, ln_lambda=2.0, name='weight'):
        super(InfinityNorm, self).__init__()
        self.module = module
        self.name = name
        self.ln_lambda = torch.tensor(ln_lambda)
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]

        absrowsum = torch.sum(w.view(height,-1).data, axis=1)
        avg = torch.mean(absrowsum, 0)
        scale = torch.minimum(torch.ones_like(absrowsum),avg/absrowsum)
        scale = scale.unsqueeze(1).unsqueeze(1).unsqueeze(1)   
        setattr(self.module, self.name, w * scale.expand_as(w))

    def _made_params(self):
        try:
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class SpectralNorm2(nn.Module):
    def __init__(self, module, ln_lambda=2.0, name='weight'):
        super(SpectralNorm2, self).__init__()
        self.module = module
        self.name = name
        self.ln_lambda = torch.tensor(ln_lambda)
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]

        _,w_svd,_ = torch.svd(w.view(height,-1).data, some=False, compute_uv=False)
        sigma = w_svd[0]
        if self.ln_lambda > 0:
           sigma = torch.max(torch.ones_like(sigma),sigma/self.ln_lambda)
        else:
           c = getattr(self.module, self.name + "_c")
           c = nn.functional.softplus(c)
           print(c)
           sigma = torch.max(torch.ones_like(sigma),sigma/(c+torch.ones_like(c)+torch.ones_like(c)))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_bar", w_bar)

        if self.ln_lambda:
           height = w_bar.data.shape[0]
           _,wsvd,_ = torch.svd(w_bar.view(height,-1).data, some=False, compute_uv=False)
           c_bar = nn.Parameter(torch.rand(1)) #torch.max(wsvd[0]))
           c_bar.requires_grad = True
           self.module.register_parameter(self.name + "_c", c_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, ln_lambda=2.0, name='weight', power_iterations=2):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.ln_lambda = ln_lambda
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        sigma = torch.max(torch.ones_like(sigma),sigma/self.ln_lambda)
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def lowpass_F(input, w=[0.0625, 0.2500, 0.0625]):
    w = torch.tensor(w)
    filter = torch.outer(w,w)
    b,in_chns = input.shape[0], input.shape[1]
    weight = torch.broadcast_to(filter, [b,in_chns]+list(filter.shape))
    return F.conv2d(input,weight,padding=1)

def lowpass_conv(in_f, pad_size, w=[0.2, 0.6, 0.2]):
    w = torch.tensor(w)
    filter = torch.outer(w,w)
    filter = torch.broadcast_to(filter, [in_f, 1, 3, 3])
    # define a depthwise 2D conv
    conv = nn.Conv2d(in_f, in_f, kernel_size=3, stride=1, padding=pad_size, bias=False, groups=in_f)
    conv.weight.data = filter.to('cuda')
    conv.weight.requires_grad = False
    return conv


def zero_insertion_lowpass_conv(in_f, pad_size, outsize=0, w=[0.25, 0.5, 0.25]):
    w = torch.tensor(w)
    filter = torch.outer(w,w)
    filter = torch.broadcast_to(filter, [in_f, 1, 3, 3])
    # define a depthwise 2D conv
    conv = nn.Conv2d(in_f, in_f, kernel_size=3, stride=1, padding=pad_size, bias=False, groups=in_f)
    conv.weight.data = filter.to('cuda')
    conv.weight.requires_grad = False
  
    # define a transposed conv for zero-insertion
    tconv = nn.ConvTranspose2d(in_f, in_f, stride=2,padding=1,kernel_size=3,bias=False,groups=in_f)
    zeros = F.pad(torch.ones(1,1,1,1),(1,1,1,1)) # for interleaved zeros
    zeros_filter = torch.broadcast_to(zeros, [in_f, 1, 3, 3])
    tconv.weight.data = zeros_filter.to('cuda')
    tconv.weight.requires_grad = False

    upsample_module = nn.Sequential(tconv,
                                    nn.ZeroPad2d((0,1,0,1)), 
                                    conv)
    return upsample_module


def learnable_zero_insertion_lowpass_conv(in_f, pad_size, outsize=0, w=[0.25, 0.5, 0.25]):
    # define a depthwise 2D conv
    conv = nn.Conv2d(in_f, in_f, kernel_size=3, stride=1, padding=pad_size, bias=False, groups=in_f)

    # define a transposed conv for zero-insertion
    tconv = nn.ConvTranspose2d(in_f, in_f, stride=2,padding=1,kernel_size=3,bias=False,groups=in_f)
    zeros = F.pad(torch.ones(1,1,1,1),(1,1,1,1)) # for interleaved zeros
    zeros_filter = torch.broadcast_to(zeros, [in_f, 1, 3, 3])
    tconv.weight.data = zeros_filter.to('cuda')
    tconv.weight.requires_grad = False

    upsample_module = nn.Sequential(tconv,
                                    nn.ZeroPad2d((0,1,0,1)), 
                                    conv
                                   # nn.Softmax(dim=1)
                                    )
    return upsample_module


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', c=0, downsample_mode='stride', hidden=False, is_last=False):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5,
                                      preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
   
    if c!=0:
       convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
       nn.init.kaiming_uniform_(convolver.weight, a=0, mode='fan_in')
       convolver = SpectralNorm2(convolver, c)

    if hidden:
       convolver = SupermaskConv(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


def branch5x5(in_f, out_f, r, to_pad):
    return nn.Sequential(
       nn.Conv2d(in_f, r * out_f, kernel_size=1, padding=to_pad, bias=True),
       nn.Conv2d(r2 * out_f, r * out_f, kernel_size=5, padding=to_pad, bias=True)
    )

def branch3x3(in_f, out_f, r, to_pad):
    return nn.Sequential(
       nn.Conv2d(in_f, r * out_f, kernel_size=1, padding=to_pad, bias=True),
       nn.Conv2d(r2 * out_f, r * out_f, kernel_size=3, padding=to_pad, bias=True)
    )

def inception_block(in_f, out_f, bias=True, pad='zero', downsample_mode='stride'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    
    r1, r2, r3, r4 = 4, 2, 8, 2

    branch1x1 = nn.Conv2d(in_f, r1 * out_f, kernel_size=1, padding=to_pad, bias=bias)
   
    branch5x5_ = branch5x5(in_f, out_f, r2, to_pad)

    branch3x3_ = branch3x3(in_f, out_f, r3, to_pad)
  
    branch_pool_1 = nn.Conv2d(in_f, r4 * out_f, kernel_size=1, stride=stride, padding=to_pad, bias=bias)

# Added by Jiang
# Seperate upsampler into zero-insertion and LPF parts. 
class InsertZeros(nn.Module):
    """
    """
    def __init__(self, up_x, up_y, gain=1.0):
        super(InsertZeros, self).__init__()
        self.upx = up_x
        self.upy = up_y
        self.gain = gain

    def forward(self, x):
        b = x.size()[0]
        c = x.size()[1]
        h = x.size()[2]
        w = x.size()[3]
        x = x.reshape([b, c, h, 1, w, 1])
        x = F.pad(x, [0, self.upx - 1, 0, 0, 0, self.upy - 1])
        x = x.reshape([b, c, h * self.upx, w * self.upy])
        x = x * self.gain
        return x     

def lowpass_conv3(num_ch, w, pad_size='same', pad_mode='zeros', gain=1.0):
    # kernel size
    k_size = len(w)
    # Convert 1D LPF coefficients to 2D
    f_2d_coeff = torch.outer(w,w)
    f_weights = torch.broadcast_to(f_2d_coeff, [num_ch, 1, k_size, k_size])
    # define a depthwise 2D conv
    print(f'kernel_size:{k_size} padding_size:{pad_size} padding_mode:{pad_mode}')
    conv = nn.Conv2d(num_ch, num_ch, kernel_size=k_size, stride=1, padding=pad_size, padding_mode=pad_mode, bias=False, groups=num_ch)
    f_weights = f_weights * gain
    #conv.weight.data = f_weights.to('cuda')
    conv.weight.data = f_weights
    conv.weight.requires_grad = False
    return conv     
    
