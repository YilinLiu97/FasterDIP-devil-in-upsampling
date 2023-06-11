import os
import numpy as np
import h5py
import itertools
import scipy.io as sio
import random
import decimal
import time

from mrrt.mri.operators import MRI_Operator
from mrrt.mri import mri_dcf_pipe
from mrrt.mri.coils import coil_pca_noncart, calculate_csm_inati

use_gpu = False
if use_gpu:
    import cupy
    xp = cupy
else:
    xp = np

data_root = '/mnt/yaplab/data/yilinliu/datasets/MRF-DIP'
dict_PATH = '/mnt/yaplab/data/yilinliu/datasets/MRF-DIP'
meas_PATH = data_root

# Check the kspace-image conversion
'''
f = h5py.File(os.path.join(path, 'kspace_144.h5'), 'r')
print(f'f.keys()={list(f.keys())}')

kr = np.asarray(f['real'], dtype=np.float32)
ki = np.asarray(f['imag'], dtype=np.float32)
kdata = kr + 1j * ki

kdata_tensor = torch.from_numpy(np.stack((kr, ki), -1))
MRFimg_coils = np.fft.fftshift(np.fft.fftshift(np.fft.ifft(np.fft.ifft(np.fft.fftshift(np.fft.fftshift(kdata_tensor,1),2),axis=1),axis=2), 2), 1)
MRFimg = np.sqrt(np.sum(MRFimg_coils * np.conj(MRFimg_coils)))

save = h5py.File('MRFimg.h5', 'w')
save.create_dataset('real', data=MRFimg[...,0])
save.create_dataset('imag', data=MRFimg[...,1])
save.close()
 
'''
def save(d, filename='save.h5', complex=False):
    f = h5py.File(os.path.join(filename), 'w')
    pts = d.shape[-1] // 2
    if complex:
        f.create_dataset('imMRF_generated/real', data=d[...,:pts])
        f.create_dataset('imMRF_generated/imag', data=d[...,pts:])
    else:
        mag = np.abs(d[...,:pts] + 1j * d[...,pts:])
        f.create_dataset('mag', data=mag)
    f.close()

def mrf_ifft(kspace_mrf=None):
    """
    kdata: (nt, N, N, nc) complex
    """
    if kspace_mrf is None:
       kf = h5py.File(os.path.join(meas_PATH, 'kspace_144.h5'), 'r')
       kr, ki = kf['real'][:], kf['imag'][:]
       kspace_mrf = kr + 1j * ki

    MRFimg_multicoil = np.fft.fftshift(np.fft.fftshift(np.fft.ifft(np.fft.ifft(np.fft.fftshift(np.fft.fftshift(kspace_mrf,1),2),axis=1),axis=2), 2), 1)
    return MRFimg_multicoil    

def cal_m0(imMRF=None):
    """
    imMRF: (nt, N, N, nc) complex
    """
    if imMRF is None:
       imMRF = mrf_ifft()
    mag_multicoil = np.abs(imMRF)
    mag_comb = np.sqrt(np.sum(mag_multicoil**2, -1))
    return np.mean(mag_comb, 0)

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)
           
def get_valid_fps(dict, pos_table, t1_pos, t2_pos, time_pts):
    c_pos = pos_table[t1_pos, t2_pos]
    if c_pos == -999999 or c_pos < 0: # invalid positions
        return np.zeros(time_pts)
    return dict[c_pos, :]

def inverse_DM(tissue_out, pos_table, t1v, t2v, dict, time_pts):
   time_pts *= 2
   t1_out, t2_out = tissue_out[...,0], tissue_out[...,1]

   w, h = t1_out.shape[-2:]
   
   t1_values_flat, t2_values_flat = t1_out.flatten(), t2_out.flatten()
      
   t1_idx_flat = np.searchsorted(t1v, t1_values_flat)
   t2_idx_flat = np.searchsorted(t2v, t2_values_flat)
   
   
   fp_out = np.zeros((w, h,time_pts))
   fp_flat = np.reshape(fp_out, (-1, time_pts))
   for i, (t1, t2, t1_pos, t2_pos) in enumerate(zip(t1_values_flat, t2_values_flat, t1_idx_flat, t2_idx_flat)):
        if (t1 == 0) and (t2 == 0):
            fp_flat[i, :] = torch.zeros(time_pts)
            continue
        elif (t1 % 1 !=0) or (t2 % 1 != 0): # if any not whole number
            print("Used bilinear interp!!!!!!!!")
            x1, y1 = t1_pos-1, t2_pos-1
            x2, y2 = t1_pos, t2_pos
            x1v, y1v = t1v[x1], t2v[y1]
            x2v, y2v = t1v[x2], t2v[y2]
            #print(f"x:{t1_pos}, y:{t2_pos}, x1: {x1}, x2: {x2}, y1:{y1}, y2:{y2}, t1v: {t1}, t2v: {t2}")
            x1y1 = np.asarray(get_valid_fps(dict, pos_table, x1, y1, time_pts))
            x1y2 = np.asarray(get_valid_fps(dict, pos_table, x1, y2, time_pts))
            x2y1 = np.asarray(get_valid_fps(dict, pos_table, x2, y1, time_pts))
            x2y2 = np.asarray(get_valid_fps(dict, pos_table, x2, y2, time_pts))
            
            neighbors = list(((x1v,y1v,x1y1), (x1v,y2v,x1y2), (x2v,y1v,x2y1), (x2v,y2v,x2y2)))
            
            fp_flat[i, :] = bilinear_interpolation(t1, t2, neighbors)
        else:
            print("--- Fine ---")
            fp_flat[i, :] = get_valid_fps(dict, pos_table, t1_pos, t2_pos, time_pts)
           
   return np.reshape(fp_flat, (w, h, -1))
   
def DM(MRFimg, Nex, dict, r, save_name, our_m0=None):
    """
    MRFimg, dict: need to be complex
    """
    if MRFimg.shape[1] != MRFimg.shape[2]:
       MRFimg = MRFimg.transpose(2,0,1) # (t, N, N)
       
    dict = dict[...,:Nex] + 1j * dict[...,Nex:]
    
    if use_gpu:
        MRFimg = cupy.asarray(MRFimg)
        dict = cupy.asarray(dict)
        
    N = MRFimg.shape[1]
    dict = dict[:,:Nex]
    print("dict: ", dict.shape)
    print("MRFimg: ", MRFimg.shape)
    MRFimg = MRFimg[:Nex,:,:]
    MRFimg = MRFimg.reshape((Nex,N*N), order='F')
    MRFimg = MRFimg.transpose()
    print("MRFimg: ", MRFimg.shape)
    MRFimgnorm = np.zeros((MRFimg.shape[0],1),dtype=np.float32)
    MRFimgnorm[:,0] = xp.sqrt(xp.sum(MRFimg * xp.conj(MRFimg),axis=1))

    # dictnorm = sqrt(sum(dict(trange,:).*conj(dict(trange,:))));
    dictnorm = np.zeros((1,dict.shape[0]),dtype=np.float32)
    dictnorm[0,:] = xp.sqrt(xp.sum(dict * xp.conj(dict),axis=1))
    normAll = xp.matmul(MRFimgnorm,dictnorm)

    # perform inner product
    #innerProduct = conj(xx)*dict(trange,:)./normAll; clear normAll
    innerproduct = xp.matmul(xp.conj(MRFimg),dict.transpose())
    innerproduct = np.abs(innerproduct) / normAll

    indexm = xp.argmax(innerproduct,axis=1)

    # extract T1 and T2 maps
    t1map = r[0,indexm[:]]
    t1map = t1map.reshape((N,N),order='F')
    t2map = r[1,indexm[:]]
    t2map = t2map.reshape((N,N),order='F')

    # calculate proton density map
    m0map = np.zeros((N*N),dtype=np.float32)

    for i in range(0,indexm.shape[0]):
        dictCol = dict[indexm[i],:]
        print("dictCol:", dictCol.shape)
        tempdictCol = dictCol.conj()/sum(dictCol.conj()*dictCol)
        print("tempdictCol: ", tempdictCol.shape)
        m0map[i] = abs(sum(tempdictCol*MRFimg[i,:]))
        print("m0map[i]:", m0map[i].shape)
    m0map = m0map.reshape((N,N),order='F')

    f=h5py.File(os.path.join(dict_PATH, f"{save_name}.h5"),'w')
    f.create_dataset('t1', data=t1map)
    f.create_dataset('t2', data=t2map)
    f.create_dataset('m0', data=m0map)
    if our_m0 is not None:
       f.create_dataset('our_m0', data=our_m0)
    f.close()


def prepare_dictionary(time_pts):
    MRFDict_filename = os.path.join(dict_PATH, 'dict.mat')
    f = h5py.File(MRFDict_filename, 'r')
    print(f'dict.keys()={list(f.keys())}')
    dict = np.asarray(f['dict'])
    dict_r = np.asarray(dict['real'], dtype=np.float32)[:,:time_pts]
    dict_i = np.asarray(dict['imag'], dtype=np.float32)[:,:time_pts]
    dict = np.concatenate((dict_r, dict_i), -1)
    r = np.asarray(f['r']) # (2, 13123)
    print(f'dict.shape:{dict.shape}, tissues:{r.shape}')
    
    t1v = np.unique(r[0,...])
    t2v = np.unique(r[1,...])

    c = list(itertools.product(t1v, t2v)) # (13674, 2) => (258 x 53, 2)

    r2 = r.transpose() # (2, 13123)
    r3 = [tuple(i) for i in r2] # convert to array of tuples
    invalid = set(c) - set(r3) # 13674 - 13123 = 551
    invalid_list = list(invalid)
    
    for i in range(len(invalid)):
        idx = c.index(invalid_list[i])
        c[idx] = 'NAN'
        
    tissue_table = np.reshape(c, [len(t1v), len(t2v)]) # (258, 53, 2)
    
    pos_table = np.empty((len(t1v), len(t2v)))
    for i in range(len(t1v)):
        for j in range(len(t2v)):
            if tissue_table[i,j] != 'NAN':
                idx = r3.index(tissue_table[i,j])
                pos_table[i,j] = idx
            else:
                pos_table[i,j] = -999999
                
    pos_table = np.asarray(pos_table, dtype='int') # (258, 53, 2)
    
    print(f"Inverse DM is ready, with tissue table: {tissue_table.shape} and idx table: {pos_table.shape}")
    return tissue_table, pos_table, t1v, t2v, dict, r

# Main
time_pts = 144
save_name_FP = f'save_fingerprints_{time_pts}'
save_name_DM = f'generated_DM_{time_pts}'
tissue_table, pos_table, t1v, t2v, dict, r = prepare_dictionary(time_pts)

'''
t1_pred = np.random.uniform(60,5000,(30,256,256))
t2_pred = np.random.uniform(10,500,(30,256,256))
'''

tf = h5py.File(os.path.join(data_root,'patternmatching_2304.mat'),'r')
#cf = sio.loadmat(os.path.join(data_root,'cmap.mat'))

t1gt = tf['t1big'][:]
t1_pred = np.clip(t1gt + np.random.uniform(0,0.6, t1gt.shape), 60, 5000)
t2gt = tf['t2big'][:]
t2_pred = np.clip(t2gt + np.random.uniform(0,0.6, t2gt.shape), 10, 500)
m0 = tf['m0big'][:]
our_m0 = cal_m0()
#cmap = cf['cmap'][:]

tissue_out = np.stack((t1_pred, t2_pred), -1)

start_time = time.time()
final = inverse_DM(tissue_out, pos_table, t1v, t2v, dict, time_pts)
print(f"Inverse DM with {time_pts} time points  taken --- {time.time() - start_time} --- seconds")

save(final* m0[...,np.newaxis], os.path.join(dict_PATH, f'{save_name_FP}.h5'), complex=True)
print(f"Finished! Final is: {final.shape} saved in {dict_PATH}")

print(f"Now doing DM using the generated fingerprints...")
DM(final, time_pts, dict, r, save_name_DM, our_m0=None)

