"""
Dynamic Channel Pruning: Feature Boosting and Suppression
"""
# -*- coding=utf-8 -*-

__all__ = [
    'ChannelPruning'
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_in_channels(m: nn.Module) -> Optional[int]:
    if isinstance(m, nn.Sequential):
        for sub in m:
            if (c := get_in_channels(sub)) is not None:
                return c
        return None
    elif isinstance(m, nn.Linear):
        return m.in_features
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return m.in_channels
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return m.num_features
    else:
        return None


def get_out_channels(m: nn.Module) -> Optional[int]:
    if isinstance(m, nn.Sequential):
        res = None
        for sub in m:
            if (c := get_in_channels(sub)) is not None:
                res = c
        return res
    elif isinstance(m, nn.Linear):
        return m.out_features
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return m.out_channels
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return m.num_features
    else:
        return None


class ChannelPruning(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.contents = nn.Sequential(*args, **kwargs)
        self.in_channels = get_in_channels(self.contents)
        self.out_channels = get_out_channels(self.contents)
        assert self.in_channels is not None
        assert self.out_channels is not None
        self.rate = 1.0
        self.loss = None
        self.gate = nn.Linear(in_features=self.in_channels,
                              out_features=self.out_channels,
                              bias=True)
        nn.init.constant_(self.gate.bias, 1)
        nn.init.kaiming_normal_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.contents(x)
        if (k := int(self.out_channels * (1.0 - self.rate))) > 0:
            s = F.adaptive_avg_pool2d(torch.abs(x), (1, 1)).view(x.size()[0], -1)
            g = F.relu(self.gate(s))
            i = (-g).topk(k, 1)[1]
            t = g.scatter(1, i, 0)
            t = t / torch.sum(t, dim=1).unsqueeze(1) * self.out_channels
            y = y * t.unsqueeze(2).unsqueeze(3)
            if self.training:
                self.loss = torch.norm(g, 1)
            else:
                self.loss = None
        return y
