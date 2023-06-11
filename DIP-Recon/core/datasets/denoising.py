import torch
from torch.utils.data import Dataset

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils.denoising_utils import *
from utils.common_utils import *

class noisyImages(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        files.sort()

        imsize = args.imsize if len(args.imsize) != 1 else -1  #if 'DIV2K' not in args.folder_path else [512,512]
        PLOT = False
        sigma = args.noise_sigma
        sigma_ = sigma/255.

        self.samples = []
        targets = 0
        print('>>> images for testing...')
        for i in range(len(files)):
          fname = files[i]        
          if args.num_scales== 8 and args.model_type == 'DIP_2_scaled' and 'House' in fname:
           continue         
          print(f"{i}: {fname}")
          img_pil = crop_image(get_image(os.path.join(args.folder_path, fname), imsize)[0], d=32)
          img_np = pil_to_np(img_pil)
          print(img_np.shape)
          print(f"min: {img_np.min()}, max: {img_np.max()}")
          img_noisy_np = get_noisy_image(img_np, sigma_, scale=args.poisson_scale)   

          if img_noisy_np.shape[0] == 1:
             save_image(f"{args.save_path}/noisy_{fname}", img_noisy_np)
          else:
             plt.imsave(f"{args.save_path}/noisy_{fname}", img_noisy_np.transpose(1,2,0))

          self.samples.append(
             {'target_img': img_noisy_np,
              'gt': img_np, 
              'filename': fname
             })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
        
class polyU(Dataset):
    def __init__(self, args):

        if args.dataset == 'polyu':
           args.folder_path = '/mnt/yaplab/data/yilinliu/datasets/PolyU-Real-World-Noisy-Images-Dataset/noisy_test'
           gt_path ='/mnt/yaplab/data/yilinliu/datasets/PolyU-Real-World-Noisy-Images-Dataset/clean_test'
        else:
           raise ValueError('dataset %s not included yet' % args.dataset)
           
        f1 = os.listdir(args.folder_path)
        f1.sort()
        
        f2 = os.listdir(gt_path)
        f2.sort()

        imsize = args.imsize if len(args.imsize) != 1 else -1

        self.samples = []
        targets = 0

        for i, (noisy, clean) in enumerate(zip(f1, f2)):
          print(f"{i}: {noisy} <=> {clean}")
          img_pil = crop_image(get_image(os.path.join(args.folder_path, noisy), imsize)[0], d=32)
          img_noisy_np = pil_to_np(img_pil)
          print(img_noisy_np.shape)
          
          gt_pil = crop_image(get_image(os.path.join(gt_path, clean), imsize)[0], d=32)
          gt_np = pil_to_np(gt_pil)
          
          if img_noisy_np.shape[0] == 1:
             save_image(f"{args.save_path}/{noisy}", img_noisy_np)
          else:
             plt.imsave(f"{args.save_path}/{noisy}", img_noisy_np.transpose(1,2,0))

          self.samples.append(
             {'target_img': img_noisy_np,
              'gt': gt_np,
              'filename': noisy
             })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SIDD(Dataset):
    def __init__(self, args):

        if args.dataset == 'SIDD':
           args.folder_path = '/mnt/yaplab/data/yilinliu/datasets/Real_World_Noisy/SIDD_Small_sRGB_Only/Data'
        else:
           raise ValueError('dataset %s not included yet' % args.dataset)

        imsize = args.imsize if len(args.imsize) != 1 else -1

        self.samples = []

        print('>>> images for testing...')
        for path, subdirs, files in os.walk(args.folder_path):
            for file in files:
                if file.endswith('.PNG'):
                   print(file)
                   file = Path(os.path.join(path, file))
                   file.rename(file.with_suffix('.png'))

        for path, subdirs, files in os.walk(args.folder_path):
            for ID in subdirs:
                if 'NOISY' or 'GT' in file:
                   noisy_fname = os.path.join(path, ID, 'NOISY_SRGB_010.png')                  
                   gt_fname = Path(os.path.join(path, ID, 'GT_SRGB_010.png'))
                   
                   img = get_image(noisy_fname, imsize)[0]
                   print(f"original image: {pil_to_np(img).shape}")
                   img_noisy_pil = crop_image(img, d=32)
#                   img_noisy_pil = crop_image(get_image(noisy_fname, imsize)[0], d=32)
                   img_noisy_np = pil_to_np(img_noisy_pil)
                   print(f"after: {img_noisy_np.shape}")
                   fname = f"{ID}_noisy"
                   print(f"{fname}:{img_noisy_np.shape}")
                
                   img_pil = crop_image(get_image(gt_fname, imsize)[0], d=32)
                   img_np = pil_to_np(img_pil)
                
                   if img_noisy_np.shape[0] == 1:
                      save_image(f"{args.save_path}/{fname}.png", img_noisy_np)
                   else:
                      plt.imsave(f"{args.save_path}/{fname}.png", img_noisy_np.transpose(1,2,0))

                   print(type(img_np))
                   self.samples.append(
                       {'target_img': img_noisy_np,
                        'gt': img_np,
                        'filename': fname
                       })
                else:
                   raise ValueError('Neither Noisy or GT is found.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
