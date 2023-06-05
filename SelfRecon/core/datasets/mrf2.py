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
            if "combined" in filename:
               f = h5py.File(args.folder_path + filename, 'r')
               rec_real, rec_imag = f['real'][:], f['imag'][:]
               print("rec_real: ", rec_real.shape)
               slice_ksp = rec_real + 1j * rec_imag
               print("slice_ksp: ", slice_ksp.shape)
            else:
               continue
            print(("Reconstruct the %s") % (filename))
                        
            mrf_frames = np.stack([rec_real, rec_imag], -1)
            slice_ksp_torchtensor = mri_fft2(torch.from_numpy(mrf_frames))

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



