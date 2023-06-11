import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Implemented as a layer of conv.
    Input: tensor [B, C, H, W]
    Output: tensor [B,C,H/stride,W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)


class LayerNormChannel(nn.Module):
    """
    only for channel dimension.
    Input: tensor [B, C, H, W]
    """
    def __init__(self, num_chans, esp=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_chans)) # gamma
        self.bias = nn.Parameter(torch.zeros(num_chans)) #beta
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True) # variance across all channels
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

