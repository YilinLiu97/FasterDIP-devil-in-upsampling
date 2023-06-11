import torch
import torch.nn
from torch.autograd import Variable

from utils.mri_utils import *
from utils.mrf_utils import *
from utils.common_utils import *
from utils.sr_utils import *

def mri_forwardm(img, mask, downsampler=None, csm=None):
    # img has dimension (2*num_slices, x,y)
    # output has dimension (1, num_slices, x, y, 2)
    mask = mask[0]

    s = img.shape
    ns = int(s[1] / 2)  # number of slices
    fimg = Variable(torch.zeros((s[0], ns, s[2], s[3], 2))).type(dtype)
    for i in range(ns):
        fimg[0, i, :, :, 0] = img[0, 2 * i, :, :]
        fimg[0, i, :, :, 1] = img[0, 2 * i + 1, :, :]
    Fimg = mri_fft2(fimg)  # dim: (1,num_coils,x,y,2)
   
    for i in range(ns):
        Fimg[0, i, :, :, 0] *= mask
        Fimg[0, i, :, :, 1] *= mask
    return Fimg

def mrf_forwardm(tmaps, dict, csm, m0map, kmask, tmask=None):
    """
    Inputs:
    tmaps: (1, 2, N, N). already coils combined.
    csm: (1, N, N, nc)
    kmask: (1, nt, N, N, nc)
    tmask: (1, N, N)
    m0map: (1, N, N, 2)

    Return: (1, nt, N, N, nc, 2)
    """
#    if tmask is not None:
#       print("used tmask!!!!!!!")
#       tmaps = tmaps * tmask.unsqueeze(1)
    # retreive fingerprints from dict
    fps = inverse_DM(tmaps, dict, tmask) # (nt, N, N, 2)
    fps, m0map = torch.view_as_complex(fps), torch.view_as_complex(m0map)
    scaled = fps * m0map
    scaled = torch.stack((scaled.real, scaled.imag), -1)
    '''
    save_fps = h5py.File('fps_recon.h5', 'w')
    save_fps.create_dataset('real', data=scaled[...,0].data.cpu().numpy())
    save_fps.create_dataset('imag', data=scaled[...,1].data.cpu().numpy())
    save_fps.close()
    '''
#    scaled = torch.stack((fps[...,0]*m0map[...,0] - m0map[...,1]*m0map[...,1], fps[...,0]*m0map[...,1] + fps[...,1]*m0map[...,0]), -1) #fps * m0map.unsqueeze(-1)
    kspace_fps = mrf_fft2(scaled, csm) # (1, nt, N, N, nc, 2)
    Fimg = kspace_fps * kmask.unsqueeze(-1)    

    print("Fimg output: ", torch.linalg.norm(Fimg))   
    return Fimg

def denoising_forwardm(img, mask=None, downsampler=None):
    return img

def inpainting_forwardm(img, mask, downsampler=None):
    assert mask is not None
    return img * mask

def sr_forwardm(img, mask, downsampler):
    return downsampler(img)
