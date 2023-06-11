import torch
from torch.utils.data import Dataset

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.sr_utils import *
from utils.common_utils import save_image
from models.downsampler import Downsampler

class blurImages(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path) 
        files.sort()

        imsize = args.imsize if len(args.imsize) != 1 else -1  #if 'DIV2K' not in args.folder_path else [512,512]
        PLOT = False
        factor = args.sr_factor
        enforce_div32 = 'CROP'

        self.samples = []
        targets = 0
        print('>>> images for super resolution...')
        for i in range(len(files)):
          fname = files[i]        
          print(fname)
          imgs = load_LR_HR_imgs_sr(os.path.join(args.folder_path, fname), imsize, factor, enforce_div32)
          imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])
      
          save_image(f"{args.save_path}/bicubic_{fname}", imgs['bicubic_np'])
          save_image(f"{args.save_path}/LR_{fname}", imgs['LR_np'])
   
          self.samples.append(
             {'target_img': imgs['LR_np'], 
              'gt': imgs['HR_np'],
              'bicubic_img': imgs['bicubic_np'],
              'nearest_img': imgs['nearest_np'],
              'filename': fname
             })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


