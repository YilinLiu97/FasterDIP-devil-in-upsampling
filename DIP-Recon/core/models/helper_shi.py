import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter


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
    return MeanOnlyBatchNorm(num_features)
    # return nn.BatchNorm2d(num_features)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, ln_lambda=2.0, name='weight'):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.ln_lambda = torch.tensor(ln_lambda)
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):

        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]

        _, w_svd, _ = torch.svd(w.view(height, -1).data, some=False, compute_uv=False)
        sigma = w_svd[0]
        sigma = torch.max(torch.ones_like(sigma), sigma / self.ln_lambda)
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

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def conv(in_f, out_f, kernel_size=3, ln_lambda=2, stride=1, bias=True, pad='zero'):
    downsampler = None
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    nn.init.kaiming_uniform_(convolver.weight, a=0, mode='fan_in')
    if ln_lambda > 0:
        convolver = SpectralNorm(convolver, ln_lambda)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


def get_kernel(kernel_width=5, sigma=0.5):
    kernel = np.zeros([kernel_width, kernel_width])
    center = (kernel_width + 1.) / 2.
    sigma_sq = sigma * sigma

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):
            di = (i - center) / 2.
            dj = (j - center) / 2.
            kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
            kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)

    kernel /= kernel.sum()

    return kernel


class gaussian(nn.Module):
    def __init__(self, n_planes, kernel_width=5, sigma=0.5):
        super(gaussian, self).__init__()
        self.n_planes = n_planes
        self.kernel = get_kernel(kernel_width=kernel_width, sigma=sigma)

        convolver = nn.ConvTranspose2d(n_planes, n_planes, kernel_size=5, stride=2, padding=2, output_padding=1,
                                       groups=n_planes)
        convolver.weight.data[:] = 0
        convolver.bias.data[:] = 0
        convolver.weight.requires_grad = False
        convolver.bias.requires_grad = False

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            convolver.weight.data[i, 0] = kernel_torch

        self.upsampler_ = convolver

    def forward(self, x):
        x = self.upsampler_(x)

        return x
