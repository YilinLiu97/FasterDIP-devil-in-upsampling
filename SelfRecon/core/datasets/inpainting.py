import torch
from torch.utils.data import Dataset

import os
import numpy as np

from utils.inpainting_utils import *

class inpaintImages(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        files.sort()

        dim_div_by = 64
        imsize = args.imsize if len(args.imsize) != 1 else -1

        self.samples = []
        for i in range(1): #len(files)):
          #  imsize = img_np.shape[-2:]
            if args.mask_type == 'bernoulli':
                fname = files[i]
                img_pil, img_np = get_image(os.path.join(args.folder_path, fname))
                img_mask_pil = get_bernoulli_mask(img_pil, args.zero_fraction)
                img_mask_np = pil_to_np(img_mask_pil)
                #print(f"{fname}: {img_np.shape}")
            elif args.mask_type == 'kate_mask':
                fname = 'kate.png'
                img_pil, img_np = get_image(os.path.join(args.folder_path, 'kate/kate.png'))
                img_mask_pil, img_mask_np = get_image(os.path.join(args.folder_path, 'kate/kate_mask.png'), imsize)
            elif args.mask_type == 'text':
                fname = files[i]
                img_pil, img_np = get_image(os.path.join(args.folder_path, fname))
                img_mask_pil = get_text_mask(img_pil)
                img_mask_np = pil_to_np(img_mask_pil)
            elif args.mask_type == 'large':
                fname = 'library.png'
                img_pil, img_np = get_image(os.path.join(args.folder_path, 'library.png'))
                img_mask_pil, img_mask_np = get_image(os.path.join(args.folder_path, 'library_mask.png'), imsize)
             
            elif args.mask_type == 'small':
                fname = 'vase.png'
                img_pil, img_np = get_image(os.path.join(args.folder_path, 'vase.png'))
                img_mask_pil, img_mask_np = get_image(os.path.join(args.folder_path, 'vase_mask.png'), imsize)
            else:
                raise NotImplementedError("No such mask type.")

            img_mask_pil = crop_image(img_mask_pil, dim_div_by)
            img_pil      = crop_image(img_pil,      dim_div_by)
          
            img_np      = pil_to_np(img_pil)
            img_mask_np = pil_to_np(img_mask_pil)

#            img_mask_np = img_mask_np.transpose(0,2,1)
   
            print(f"img_np: {img_np.shape}, img_mask_np: {img_mask_np.shape}")
            if img_np.shape[0] == 1:
               save_image(f"{args.save_path}/corrupted_{fname}", img_np * img_mask_np)
            else:
               plt.imsave(f"{args.save_path}/corrupted_{fname}", (img_np * img_mask_np).transpose(1,2,0))

            self.samples.append({
                'target_img': img_np * img_mask_np,
                'mask': img_mask_np,
                'gt': img_np,
                'filename': fname
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

