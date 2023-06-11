from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

__all__ = ['ResNet20_cifar', 'ResNet56_cifar', 'ResNet_test']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # Gate layers
        self.fc1 = nn.Conv2d(in_planes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        # initialize the bias of the last fc for 
        # initial opening rate of the gate of about 85%
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        self.gs = GumbleSoftmax()
        self.gs.cuda()

    def forward(self, x, temperature=1):
        # Compute relevance score
        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1bn(self.fc1(w)))
        w = self.fc2(w)
        # Sample from Gumble Module
        w = self.gs(w, temp=temperature, force_hard=True)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + out * w[:,1].unsqueeze(1)
        out = F.relu(out)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out, w[:, 1]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        return out

    def forward_sss(self, x, drop):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out *= drop
        out += residual

        return out

    def forward_test(self, x, drop):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        if drop == 0:
            out = residual
        else:
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)

            out = self.bn3(out)
            out = self.relu(out)
            out = self.conv3(out)

            out *= drop
            out += residual

        return out


class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.inplanes = 16

        # declare lambda array
        num_block = 0
        for i in num_blocks:
            num_block += i
        print ('block number:', num_block)
        self.lambda_block = Parameter(torch.ones(num_block))

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
#        self.avgpool = nn.AvgPool2d(8)
#        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

 #       x = self.avgpool(x)
 #       x = x.view(x.size(0), -1)
 #       x = self.fc(x)
        return x

    def forward_sss(self, x):
        x = self.conv1(x)
        num_block = 0
        for block in self.layer1:
            x = block.forward_sss(x, self.lambda_block[num_block])
            num_block += 1
        for block in self.layer2:
            x = block.forward_sss(x, self.lambda_block[num_block])
            num_block += 1
        for block in self.layer3:
            x = block.forward_sss(x, self.lambda_block[num_block])
            num_block += 1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_test(self, x):
        x = self.conv1(x)
        num_block = 0
        for block in self.layer1:
            x = block.forward_test(x, self.lambda_block[num_block])
            num_block += 1
        for block in self.layer2:
            x = block.forward_test(x, self.lambda_block[num_block])
            num_block += 1
        for block in self.layer3:
            x = block.forward_test(x, self.lambda_block[num_block])
            num_block += 1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet20_cifar():
    return ResNet_cifar(Bottleneck, [2,2,2])

def ResNet56_cifar():
    return ResNet_cifar(Bottleneck, [6,6,6])

def ResNet101_cifar():
    return ResNet_cifar(Bottleneck, [11,11,11])

def ResNet164_cifar():
    return ResNet_cifar(Bottleneck, [18,18,18])

def ResNet_test():
    return ResNet_cifar(Bottleneck, [4,5,2])

model = ResNet20_cifar()
print(model)
s  = sum([np.prod(list(p.size())) for p in model.parameters()]); 
print ('Number of params: %d' % s)
