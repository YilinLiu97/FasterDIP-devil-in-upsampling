import torch
from torch.utils.data import Dataset

import os
import numpy as np
import h5py

from utils.mri_utils import *

class mrfData(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        self.samples = []
        # 
        for filename in files:
            if filename.endswith('.mat') and 'MRFdata' in filename:
               print(("Reconstruct the %s") % (filename))                        
               file = h5py.File(os.path.join(args.folder_path, filename), 'r')
               mrf_frames = file['MRFdata']['kspace'] # (2304, 1)

               str_id = len(mrf_frames) // 2 # get the central slice
               st = mrf_frames[str_id][0]
               slice = f[st] # n, nc: (1142, 32)
               slice_ksp_torchtensor = mri_fft2(torch.from_numpy(slice))

            masked_kspace, mask, mask1d, mask2d = get_mask(slice_ksp_torchtensor, slice_ksp, factor=args.ac_factor, cent=args.center_frac)
         
            self.samples.append({
                'slice_ksp': masked_kspace,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'mask': mask,
                'mask1d': mask1d,
                'mask2d': mask2d,
                'filename': filename 
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



