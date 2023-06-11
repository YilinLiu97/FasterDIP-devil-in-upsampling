import torch
from torch.utils.data import Dataset

import os
import numpy as np
import h5py

from utils.mri_utils import *
from utils.common_utils import *

class mrfData(Dataset):
    def __init__(self, args):

        files = os.listdir(args.folder_path)
        self.samples = []
        
        f = h5py.File(os.path.join(args.folder_path, 'kspace_pca_multicoil.h5'), 'r') # (nt, N, N, nc)
        d_real, d_imag = f['real'][::args.skip_frames], f['imag'][::args.skip_frames]               
        dc = d_real + 1j * d_imag        

        slice_ksp = np.stack((d_real, d_imag), axis=-1) # (nt, 256, 256, nc, 2)
        slice_ksp_torchtensor = torch.from_numpy(slice_ksp)

        kmask = (1 - (dc == 0.+0.j)) # (nt, 256, 256, nc)
        mask_tensor = torch.from_numpy(kmask)
        masked_kspace = mask_tensor.unsqueeze(-1) * slice_ksp_torchtensor

        mf = h5py.File(os.path.join(args.gt_mask_path, 'mask.mat'), 'r')
        tmask = mf['mask'][:] # (256, 256)
        
        tf = h5py.File(os.path.join(args.gt_mask_path, 'patternmatching_2304.mat'), 'r')
        t1map = tf['t1big'][:] 
        t2map = tf['t2big'][:] 

        mf = h5py.File(os.path.join(args.folder_path, 'm0_2304pnts.h5'), 'r')
        mr = mf['real'][:] 
        mi = mf['imag'][:]
        m0map = np.stack((mr, mi), -1) # (256, 256, 2)        

        cf = h5py.File(os.path.join(args.folder_path, 'cmap_pca.h5'), 'r')
        cr, ci = cf['real'][:], cf['imag'][:]
        cmap = np.stack((cr, ci), -1) # (256, 256, nc, 2)
        
        
        print(f"Reconstruct: {slice_ksp.shape} | kmask: {kmask.shape} ")
        
         
        self.samples.append({
                'slice_ksp': masked_kspace,
                'slice_ksp_torchtensor': slice_ksp_torchtensor,
                'kmask': kmask,
                'tmask': tmask,
                'm0map': m0map,
                't1map': t1map,
                't2map': t2map,
                'cmap': cmap,
                'filename': 'test_subj53_144pts'
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



