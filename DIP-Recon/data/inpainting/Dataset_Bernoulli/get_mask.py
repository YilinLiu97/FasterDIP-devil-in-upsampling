## import libs
from __future__ import print_function
from PIL import Image
import numpy as np
import h5py

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

ffname = 'peppers256'

data_root = './img/'
img_path = data_root + ffname + '.png'

img = Image.open(img_path)
img_np = pil_to_np(img)


np.random.seed(100)
img_mask_np = (np.random.random_sample(size=img_np.shape) > 0.5).astype(int)

with h5py.File('./mask/'+ ffname + '.h5', 'w') as hf:
    hf['mask'] = img_mask_np

