import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder, ViTransformerWrapper
from einops import rearrange, repeat, reduce

input = torch.rand(1,3,512,512)

def exists(val):
    return val is not None

class ViTransformerWrapper2(nn.Module):
    def __init__(
        self,
        input_dim,
        image_size,
        patch_size,
        attn_layers,
        num_classes = None,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = input_dim * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = FeedForward(dim, dim_out = num_classes, dropout = dropout) if exists(num_classes) else None

    def forward(
        self,
        img,
        return_embeddings = False
    ):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        if not exists(self.mlp_head) or return_embeddings:
            return x

        return self.mlp_head(x[:, 0])

class MiniViT2(nn.Module):

  def __init__(self, input_dim=32):

    super().__init__()

    self.patch_size = 8
    self.model = nn.Sequential(ViTransformerWrapper2(
      input_dim,
      image_size = 512,
      patch_size = self.patch_size,
      attn_layers = Encoder(
          dim = 64,
          depth = 4,
          heads = 3
      )),
      nn.Linear(64, self.patch_size**2*3))
    self.refine = nn.Sequential(
#                nn.Conv2d(3,3,1, bias=False),
                nn.Conv2d(3,3,1), nn.Sigmoid())
    
  def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

  def forward(self, x):
    p = self.patch_size
    h = w = int(x.shape[1]**.5)

    x = self.model(x)
    x = self.unpatchify(x)
    x = self.refine(x)
    return x
