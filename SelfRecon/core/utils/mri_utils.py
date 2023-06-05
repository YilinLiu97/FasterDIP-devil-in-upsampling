import torch
import numpy as np
from torch.autograd import Variable

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

dtype = torch.cuda.FloatTensor

def combine_coil_rss(multi_images, orig_shape, csm=None):
    imgs_out = []
    for img in multi_images.detach().cpu():
        imgs_out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]

    imgs_out = channels2imgs(np.array(imgs_out))
    
    imgs_out = root_sum_of_squares(torch.from_numpy(imgs_out)).numpy()
    if imgs_out.shape[0] > 320:
       s = orig_shape[-2:]
       output = crop_center(imgs_out,320,320)
    return output

def fft2_coils(data, csm=None):
    if csm is None:
       return fft2(data)
    assert csm.shape[0] == 1
    csm_r, csm_i = torch.chunk(csm, 2, dim=1)
    x_r, x_i = torch.chunk(data, 2, dim=-1)
    x1 = x_r * csm_r.unsqueeze(-1) - x_i * csm_i.unsqueeze(-1)
    x2 = x_r * csm_i.unsqueeze(-1) + x_i * csm_r.unsqueeze(-1)
    x_multicoil = torch.cat([x1, x2], -1)
    return mri_fft2(x_multicoil)

def ifft2_coils(data, csm=None):
    if csm is None:
       return ifft2(data)
    assert csm.shape[0] == 1
    csm_r, csm_i = torch.chunk(csm, 2, dim=1)
    imgs = ifft2(data)
    x_r, x_i = torch.chunk(imgs, 2, dim=-1)
    x1 = x_r * csm_r + x_i * csm_i
    x2 = x_i * csm_r - x_r * csm_i
    coil_images = torch.stack([x1, x2], -1)
    coil_combined = torch.sum(coil_images, 1)
    return coil_combined

def data_consistency(net, ni, mask1d, slice_ksp_torchtensor1, orig_shape, csm=None):
    img = net(ni.type(dtype))

    s = img.shape
    ns = int(s[1] / 2)  # number of slices
    fimg = Variable(torch.zeros((s[0], ns, s[2], s[3], 2))).type(dtype)
    ### get under-sampled measurement and put it in the Fourier representation of the reconstruction
    for i in range(ns):
        fimg[0, i, :, :, 0] = img[0, 2 * i, :, :]
        fimg[0, i, :, :, 1] = img[0, 2 * i + 1, :, :]
  
    Fimg = mri_fft2(fimg)  # dim: (1,num_slices,x,y,2)
    # ksp has dim: (num_slices,x,y)
    meas = slice_ksp_torchtensor1.data.cpu()  # dim: (1,num_slices,x,y,2)
 
    mask = mask1d[0] #torch.from_numpy(np.array(mask1d, dtype=np.uint8))
    ksp_dc = Fimg.clone()
    ksp_dc = ksp_dc.detach().cpu()
    ksp_dc[:, :, :, mask == 1, :] = meas[:, :, :, mask == 1, :]  # after data consistency block

    ### compute the inverse fourier transform of the consistency-enforced k-space
    img_dc = ifft2(ksp_dc)[0]
    out = []
    for img in img_dc.detach().cpu():
        out += [img[:, :, 0].numpy(), img[:, :, 1].numpy()]

    ### apply root sum of squares and then crop the image
    par_out_chs = np.array(out)
    par_out_imgs = channels2imgs(par_out_chs)
    prec = root_sum_of_squares(torch.from_numpy(par_out_imgs)).numpy()

    prec = crop_center(prec, 320, 320)
    return prec


def mrf_data_consistency(net, ni, mask2d, slice_ksp_torchtensor1, orig_shape, csm=None):
    img = net(ni.type(dtype))

    s = img.shape
    ns = int(s[1] / 2)  # number of slices
    fimg = Variable(torch.zeros((s[0], ns, s[2], s[3], 2))).type(dtype)
    ### get under-sampled measurement and put it in the Fourier representation of the reconstruction
    for i in range(ns):
        fimg[0, i, :, :, 0] = img[0, 2 * i, :, :]
        fimg[0, i, :, :, 1] = img[0, 2 * i + 1, :, :]
  
    Fimg = mri_fft2(fimg)  # dim: (1,num_slices,x,y,2)
    # ksp has dim: (num_slices,x,y)
    meas = slice_ksp_torchtensor1.data.cpu().float()  # dim: (1,num_slices,x,y,2)
 
    mask = mask2d[0].data.cpu() #torch.from_numpy(np.array(mask1d, dtype=np.uint8))
    ksp_dc = Fimg.clone()
    ksp_dc = ksp_dc.detach().cpu()
    print(meas.shape)    
    for i in range(ns):
        orig_1 = meas[0,i,:,:,0] * mask
        orig_2 = meas[0,i,:,:,1] * mask
        ksp_dc[0,i,:,:,0] = orig_1  # after data consistency block
        ksp_dc[0,i,:,:,1] = orig_2

    ### compute the inverse fourier transform of the consistency-enforced k-space
    img_dc = ifft2(ksp_dc)[0]
    out = []
    for img in img_dc.detach().cpu():
        out += [img[:, :, 0].numpy(), img[:, :, 1].numpy()]

    ### apply root sum of squares and then crop the image
    par_out_chs = np.array(out)
    par_out_imgs = channels2imgs(par_out_chs)
    prec = root_sum_of_squares(torch.from_numpy(par_out_imgs)).numpy()
    return prec

def get_scale_factor(net, ni, masked_kspace, batch_size=1):
    ### get norm of deep decoder output
    # get net input, scaling of that is irrelevant
    '''
        shape = [batch_size, num_channels, in_size[0], in_size[1]]
        ni = Variable(torch.zeros(shape)).type(dtype)
        ni.data.uniform_()
    '''
    # generate random image for the above net input
    out_chs = net(ni.type(dtype)).data.cpu().numpy()[0]
    out_imgs = channels2imgs(out_chs)
    out_img_tt = root_sum_of_squares(torch.tensor(out_imgs), dim=0)

    ### get norm of zero-padded image
    orig_tt = ifft2(masked_kspace)  # Apply Inverse Fourier Transform to get the complex image
    orig_imgs_tt = complex_abs(orig_tt)  # Compute absolute value to get a real image
    orig_img_tt = root_sum_of_squares(orig_imgs_tt, dim=0)
    orig_img_np = orig_img_tt.cpu().numpy()

    ### compute scaling factor as norm(output)/norm(ground truth)
    s = np.linalg.norm(out_img_tt) / np.linalg.norm(orig_img_np)
    print(s)
    return s

def get_scale_factor_mrf(net, num_channels, in_size, t1map, t2map, ni=None, batch_size=1):
    ### get norm of deep decoder output
    # get net input, scaling of that is irrelevant
    '''
    if ni is None:
        shape = [batch_size, num_channels, in_size[0], in_size[1]]
        ni = Variable(torch.zeros(shape)).type(dtype)
        ni.data.uniform_()
    '''
    # generate random image for the above net input
    out_chs = net(ni.type(dtype)).data.cpu().numpy() # batch_size > 1
    
    ### compute scaling factor as norm(output)/norm(ground truth)
    s_t1 = np.linalg.norm(out_chs[0,0,...]) / np.linalg.norm(t1map)
    s_t2 = np.linalg.norm(out_chs[0,1,...]) / np.linalg.norm(t2map)

    return s_t1, s_t2, ni

def normalize(im1,im2):
    # im1: ground truth
    # im2: reconstruction
    im1 = (im1-im1.mean()) / im1.std()
    im1 *= im2.std()
    im1 += im2.mean()
    return im1,im2
def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )
def psnr(gt, pred):
    """ Compute PSNR. """
    return compare_psnr(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), data_range=gt.max()
    )

class MaskFunc:
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    MaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


def np_to_var(img_np, dtype=torch.cuda.FloatTensor):
    '''
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Converts image in numpy.array to torch.Variable.

    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(torch.from_numpy(img_np)[None, :])

def var_to_np(img_var):
    '''
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Converts an image in torch.Variable format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]

def ksp2measurement(ksp):
    return np_to_var( np.transpose( np.array([np.real(ksp),np.imag(ksp)]) , (1, 2, 3, 0)) )

def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))

def channels2imgs(out):
    """
    :param out: real-imag stacked inputs (1,2C, H, W)
    :return: magnitude (1,C,H,W)
    """
    sh = out.shape
    chs = int(sh[0]/2)
    imgs = np.zeros( (chs,sh[1],sh[2]) )
    for i in range(chs):
        imgs[i] = np.sqrt( out[2*i]**2 + out[2*i+1]**2 )
    return imgs

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def get_mask(slice_ksp_torchtensor, slice_ksp, factor=4, cent=0.07):
    """
    :param slice_ksp_torchtensor: (coils, H, W, 2)
    :param slice_ksp: (coils, H, W)
    :param factor:
    :param cent:
    :return:
    """
    masked_kspace, mask = apply_mask(slice_ksp_torchtensor,
                                   mask_func=MaskFunc(center_fractions=[cent], accelerations=[factor]))
    mask1d = mask[0, 0, :, 0]
    mask1d[:mask1d.shape[-1] // 2 - 160] = 0
    mask1d[mask1d.shape[-1] // 2 + 160:] = 0
    mask2d = np.repeat(mask1d.numpy()[None, :], slice_ksp.shape[1], axis=0).astype(int)
    mask2d = np.pad(mask2d, ((0,), ((slice_ksp.shape[-1] - mask2d.shape[-1]) // 2,)), mode='constant')
    return masked_kspace, mask, mask1d, mask2d

def get_mask2(slice_ksp_torchtensor, slice_ksp,factor=4,cent=0.07):
    try: # if the file already has a mask
        temp = np.array([1 if e else 0 for e in f["mask"]])
        temp = temp[np.newaxis].T
        temp = np.array([[temp]])
        mask = to_tensor(temp).type(dtype).detach().cpu()
    except: # if we need to create a mask
        desired_factor = factor # desired under-sampling factor
        undersampling_factor = 0
        tolerance = 0.03
        while undersampling_factor < desired_factor - tolerance or undersampling_factor > desired_factor + tolerance:
            mask_func = MaskFunc(center_fractions=[cent], accelerations=[desired_factor])  # Create the mask function object
            masked_kspace, mask = apply_mask(slice_ksp_torchtensor, mask_func=mask_func)   # Apply the mask to k-space
            mask1d = var_to_np(mask)[0,:,0]
            undersampling_factor = len(mask1d) / sum(mask1d)

    mask1d = var_to_np(mask)[0,:,0]

    # The provided mask and data have last dim of 368, but the actual data is smaller.
    # To prevent forcing the network to learn outside the data region, we force the mask to 0 there.
    mask1d[:mask1d.shape[-1]//2-160] = 0
    mask1d[mask1d.shape[-1]//2+160:] =0
    mask2d = np.repeat(mask1d[None,:], slice_ksp.shape[1], axis=0).astype(int) # Turning 1D Mask into 2D that matches data dimensions
    mask2d = np.pad(mask2d,((0,),((slice_ksp.shape[-1]-mask2d.shape[-1])//2,)),mode='constant') # Zero padding to make sure dimensions match up
    mask = to_tensor( np.array( [[mask2d[0][np.newaxis].T]] ) ).type(dtype).detach().cpu()
    return mask, mask1d, mask2d

def apply_mask(data, mask_func = None, mask = None, seed=None):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Subsample given k-space by multiplying with a mask.
    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.
    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    if mask is None:
        mask = mask_func(shape, seed)
    return data * mask, mask

def fft(input, signal_ndim, normalized=False):
  # This function is called from the fft2 function below
  if signal_ndim < 1 or signal_ndim > 3:
    print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
    return

  dims = (-1)
  if signal_ndim == 2:
    dims = (-2, -1)
  if signal_ndim == 3:
    dims = (-3, -2, -1)

  norm = "backward"
  if normalized:
    norm = "ortho"

  return torch.view_as_real(torch.fft.fftn(torch.view_as_complex(input), dim=dims, norm=norm))

def ifft(input, signal_ndim, normalized=False):
  # This function is called from the ifft2 function below
  if signal_ndim < 1 or signal_ndim > 3:
    print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
    return

  dims = (-1)
  if signal_ndim == 2:
    dims = (-2, -1)
  if signal_ndim == 3:
    dims = (-3, -2, -1)

  norm = "backward"
  if normalized:
    norm = "ortho"

  return torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(input), dim=dims, norm=norm))

def mri_fft2(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 2 dimensional Fast Fourier Transform. It calls the fft function above to make it compatible with the latest version of pytorch.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 2-dimensional Inverse Fast Fourier Transform. It calls the ifft function above to make it compatible with the latest version of pytorch.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def complex_abs(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()

def fftshift(x, dim=None):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def roll(x, shift, dim):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)
