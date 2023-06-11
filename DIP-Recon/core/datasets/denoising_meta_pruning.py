import torch
from torch.utils.data import Dataset

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.denoising_utils import *

class metaNoisyImages(object):
    def __init__(self, args):

        files = os.listdir(args.folder_path)

        imsize = -1
        PLOT = False
        sigma = args.noise_sigma
        sigma_ = sigma/255.

        noisy_imgs, gts, filenames = [], [], []
        targets = 0
        print('>>> meta images...')
        for i in args.meta_img_idx:
          fname = files[i]
          print(fname)
          img_pil = crop_image(get_image(os.path.join(args.folder_path, fname), imsize)[0], d=32)
          img_np = pil_to_np(img_pil)
        
          img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
        #plt.imsave('/shenlab/lab_stor/yilinliu/SelfRecon/saves/denoising/ConvDecoder/F16_noisy.png', img_noisy_np.transpose(1,2,0))

          noisy_imgs.append(img_noisy_np)
          gts.append(img_np)
          filenames.append(fname)

        self.samples = {'target_imgs': torch.from_numpy(np.concatenate(noisy_imgs)), 'gts': gts, 'filenames': filenames}
 

    def __call__(self):
        return self.samples
