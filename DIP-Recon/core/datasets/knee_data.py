import torch
from torch.utils.data import Dataset

import os
import numpy as np
import h5py

from utils.mri_utils import *

class kneeData(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
     
        self.samples = []
        # get only the central slice from each subject/file
        for i in range(len(files)):
            print(files[i])
            f = h5py.File(args.folder_path + files[i], 'r')
            slicenu = f["kspace"].shape[0] // 2 - 4
            print(("Reconstruct the %d slice") % (slicenu+1))
            slice_ksp = f["kspace"][slicenu]
            print("slice_ksp: ", slice_ksp.shape)
            kdata = np.stack((slice_ksp.real, slice_ksp.imag), axis=-1)
            csm = f["csm"][:] if 'csm' in f.keys() else torch.ones([slice_ksp.shape[0]*2] + list(slice_ksp.shape[-2:]))

            orig = f["reconstruction_rss"][slicenu]

            slice_ksp_torchtensor = torch.from_numpy(kdata)
            masked_kspace, mask, mask1d, mask2d = get_mask(slice_ksp_torchtensor, slice_ksp, factor=args.ac_factor, cent=args.center_frac)
         
            undersampled_recon = self.simple_recon(masked_kspace)

            self.samples.append({
                'slice_ksp': masked_kspace,
                'orig': orig,
                'csm' : csm,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'undersampled_recon': undersampled_recon,
                'mask': mask,
                'mask1d': mask1d,
                'mask2d': mask2d,
                'filename': files[i] 
            })

    def simple_recon(self, kspace):
        multi_imgs = ifft2(kspace)
        imgs = []
        for img in multi_imgs.detach().cpu():
            imgs += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        imgs = channels2imgs(np.array(imgs))
        imgs = root_sum_of_squares(torch.from_numpy(imgs))
        imgs = crop_center(imgs, 320,320)
        return imgs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



