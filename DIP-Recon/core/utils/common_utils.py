import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os
import sys
import subprocess

import numpy as np
from numpy.fft import fft2, fftshift
from numpy.linalg import matrix_rank
import math
import scipy
from scipy.stats import wasserstein_distance
from PIL import Image
import PIL
import numpy as np
from .radialProfile import *
from .denoising_utils import *
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

dtype = torch.cuda.FloatTensor

def getOptimizer(param, args):
    if args.optimizer == 'rmsprop':
       return torch.optim.RMSprop(param, args.lr)
    elif args.optimizer == 'adam':
       return torch.optim.Adam(param, args.lr)
    elif args.optimizer == 'adamw':
       return torch.optim.AdamW(param, args.lr, args.decay)
    elif args.optimizer == 'sgd':
       return torch.optim.SGD(param, args.lr, momentum=0.9, nesterov=True)
    else:
       raise NotImplementedError("No such optimizer type.")    

def gen_noise(shape, noise_type='u'):
    """
    draw the noise input tensor
    noise_type: 'u' -- uniform, 'n' -- normal
    """
    assert noise_type == 'u' or noise_type == 'n'
    ni = Variable(torch.zeros(shape)).type(dtype)
    ni.data.uniform_() if noise_type == 'u' else ni.data.normal_()
    return ni

def get_vacant_gpu():
    com="nvidia-smi|sed -n '/%/p'|sed 's/|/\\n/g'|sed -n '/MiB/p'|sed 's/ //g'|sed 's/MiB/\\n/'|sed '/\\//d'"
    gpum=subprocess.check_output(com, shell=True)
    gpum=gpum.decode('utf-8').split('\n')
    gpum=gpum[:-1]
    for i,d in enumerate(gpum):
        gpum[i]=int(gpum[i])
    gpu_id=gpum.index(min(gpum))
    if len(gpum)==4:
        gpu_id=3-gpu_id
    return gpu_id

def std_convoluted(image, N=1):
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)
    
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    return np.sqrt((s2 - s**2 / ns) / ns)

def normalize_img(img):
    max_v = img.max()
    min_v = img.min()
    if max_v - min_v <= 0:
        return img
    return (img - min_v) / (max_v - min_v)

def rgb2gray(rgb):
    if rgb.shape[-1] != 3:
      rgb = rgb[0].permute(1,2,0)
    return torch.matmul(rgb[...,:3], torch.tensor([0.2989, 0.5870, 0.1140]).cuda())

def emd(arr1, arr2):
    '''Earth mover's distance between arr1 and arr2.'''
    
    if len(arr1.shape) == 1 and len(arr2.shape) == 1:
        dist = np.float64(wasserstein_distance(arr1, arr2))
        return dist
    
    if len(arr1.shape) == 2 and len(arr2.shape) == 1:
        arr1, arr2 = arr2, arr1
    
    if len(arr1.shape) == 1 and len(arr2.shape) == 2:
        
        dist = np.zeros((arr2.shape[0],))
        for i, ar2 in enumerate(arr2):
            dist[i] = np.float64(wasserstein_distance(arr1, ar2))
        
        return dist
    
    if len(arr1.shape) == 2 and len(arr2.shape) == 2:
        assert arr1.shape[0] == arr2.shape[1]
        
        dist = np.zeros((arr2.shape[0],))
        for i, (ar1, ar2) in enumerate(zip(arr1, arr2)):
            dist[i] = np.float64(wasserstein_distance(ar1, ar2))
        
        return dist
    
    assert False

#def rgb2gray(rgb):
    
 #   if rgb.shape[-1] != 3:
  #    rgb = rgb.transpose(0,2,3,1)
   # return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def fft_image(img):
    assert torch.is_tensor(img)
    img_gray = rgb2gray(img)
    freq = torch.fft.fftshift(torch.fft.fft2(img_gray)) #complex tensor
    return torch.stack([freq.real, freq.imag], -1)
        

def fft_mag(img, log=False):
    freq = fft_image(img) # 2-channel real-valued
    freq = torch.view_as_complex(freq)
    if log:
       return torch.log(torch.abs(freq))
    return torch.abs(freq).float()

def fft_np(img):
    assert torch.is_tensor(img) == False
    freq = np.fft.fft2(img, norm='forward')
    psd2D = np.abs(freq)**2
    psd2D_db = fftshift(np.log(psd2D) * 10)
    psd2D_db_norm = norm(psd2D_db)
    psd2D_nodc = nodc(psd2D_db_norm)
    return psd2D_nodc

def nodc(arr):
    '''Remove the DC component.'''
    
    if len(arr.shape) in (1, 3):
        arr = arr - arr.mean()
        return arr
    
    if len(arr.shape) == 2:
        arr = arr - arr.mean(-1, keepdims=True)
        return arr
    
    if len(arr.shape) == 4:
        arr = arr - arr.mean((-1, -2, -3), keepdims=True)
        return arr
    
    assert False

def norm(arr, min=0, max=1):
    '''Normalize the given array between min and max.'''
    
    arr = arr - arr.min()
    arr = arr / arr.max()
    arr = arr * (max - min)
    arr = arr + min
    return arr

def psd(image):
    '''Power spectral density of the given image.'''
    
    image_f = fft2(image, norm='forward')

    image_psd = np.abs(image_f)**2

    return fftshift(image_psd)

def db(arr):
    '''Calculate the dB of the given array element wise.'''
    
    arr_db = 10*np.log(arr)

    return arr_db

def psd_db(image):
    '''Applie first psd and then db functions.'''
    
    image_psd = psd(image)
    return db(image_psd)

def psd_db_norm(image):
    '''Applie psd, db and norm functions.'''
    
    return norm(psd_db(image))

def get_psd(img, log=False):
    img_gray = rgb2gray(img)
    img_gray = img_gray.data.cpu().numpy()
    psd2D = psd_db_norm(img_gray)
    psd1D = numpy_azimuthalAverage(psd2D)
    psd1D = norm(psd1D)
    return psd2D, psd1D 

def cal_bw(psd2d, p=0.75):
    return per_bw(psd2d, p)

def eval_general(gt, pred):
   s = compare_ssim(gt.transpose(1,2,0), pred.transpose(1,2,0), multichannel=True)
   p = compare_psnr(gt.transpose(1,2,0), pred.transpose(1,2,0))
   return {'ssim': s, 'psnr': p}

def save_fig(img, img_dir):
    img = np.asarray(normalize_img(img) * 255, dtype=np.uint8)
    img = np.asarray([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
    img = img.transpose((1, 2, 0))
    image = Image.fromarray(img, 'RGB')
    image.save(img_dir)

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(images_np, nrow=8, factor=1, interpolation='lanczos'):
    """Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np) + factor, 12 + factor))

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.show()

    return grid


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_(std=2)
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10, freq_dict=None, batch_size=1,  args=None):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [batch_size, input_depth, spatial_size[0], spatial_size[1]]
        net_input = Variable(torch.zeros(shape)).type(dtype)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        input_depth = 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = Variable(np_to_torch(meshgrid)).type(dtype)

    elif method == 'fourier':
        if freq_dict['method'] == 'log':
            freqs = freq_dict['base'] ** torch.linspace(0., freq_dict['n_freqs'] - 1, steps=freq_dict['n_freqs'])
            net_input = Variable(generate_fourier_feature_maps(freqs, spatial_size, only_cosine=freq_dict['cosine_only'])).type(dtype)
        else:
            raise ValueError
    elif method == 'image':
        img_pil = crop_image(get_image(os.path.join(args.folder_path, args.input_image_name), args.imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        net_input = np_to_torch(img_np).type(dtype)

    elif method == 'noisy_image':
        img_pil = crop_image(get_image(os.path.join(args.folder_path, args.input_image_name), args.imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_noisy_np = get_noisy_image(img_np, args.noise_sigma/255, scale=args.poisson_scale) 
        net_input = np_to_torch(img_noisy_np).type(dtype)

    else:
        assert False

    args.input_dim = net_input.shape[1]
    return net_input, args


def generate_fourier_feature_maps(net_input, spatial_size, dtype=torch.float32, only_cosine=False):
    X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
    meshgrid_np = np.concatenate([X[None, :], Y[None, :]])

    meshgrid = torch.from_numpy(meshgrid_np).permute(1, 2, 0).unsqueeze(0).type(dtype)
    vp = net_input * torch.unsqueeze(meshgrid, -1) # (1,512,512,2,8) outer product

    if only_cosine:
        vp_cat = torch.cat((torch.cos(vp),), dim=-1)
    else:
        vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1).permute(0, 3, 1, 2)


def save_grayimage(save_path, I):
#    assert I.min() == 0.
    I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(I8[0])
    img.save(save_path)

def save_image(save_path, img):
    if img.shape[0] == 1:
       save_grayimage(save_path, img)
    else:
       plt.imsave(save_path, img.transpose(1,2,0))

def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)

        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False

def network_info(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(f"Total number of params: {num_params}")

def measure_model(net,actual_size=False):
    if actual_size:
       num_params = 0
       for param in net.parameters():
           num_params += param.numel()
       return num_params
    nonzeros = 0
    for param in net.parameters():
        tensor = param.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        nonzeros += nz_count
    return nonzeros

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x) 

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def power_iteration(A, num_simulations=10):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    A = A.data
    height=A.data.shape[0]
    width = A.view(height,-1).data.shape[1]
    u = A.new(height).normal_(0, 1)
    v = A.new(width).normal_(0, 1)
    u = l2normalize(u)
    v = l2normalize(v)
    for _ in range(num_simulations):
        v.data = l2normalize(torch.mv(torch.t(A.view(height,-1)), u))
        u.data = l2normalize(torch.mv(A.view(height,-1), v))
    return u.dot(A.view(height, -1).mv(v))

def svd_rank(A):
    height=A.data.shape[0]
    return matrix_rank(A.view(height,-1))

def max_sigma(A):
    height=A.data.shape[0]
    _,w_svd,_ = torch.svd(A.view(height,-1), some=False, compute_uv=False)
    sigma = w_svd[0]
    return sigma, w_svd

def spectral_norm(model, num_iterations=10, store=False):
    norms = []
    values = []
    for layer in model.modules():
         if isinstance(layer, nn.Conv2d):
            if layer.in_channels == layer.out_channels:
               Sigma, w_svd = max_sigma(layer.weight)
               norms.append(Sigma.detach().cpu().numpy())
               values.append(w_svd.detach().cpu().numpy())               
            elif layer.out_channels == 3 or layer.out_channels == 1 or layer.in_channels == 3 or layer.in_channels == 3:
               norms.append(torch.norm(layer.weight.data).cpu().numpy())
    print(">>>>>>>>>>>>>>> svd values: ", np.array(values).shape)
    return np.array(norms), np.array(values)
        
class clipSTE(torch.autograd.Function):
    """Clip {min(Tissue value), max(Tissue value)} a real valued tensor. Backward is STE"""

    @staticmethod
    def forward(ctx, min, max, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(min)] = min
        outputs[inputs.gt(max)] = max

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        gradInput = gradOutput.clone()
        #gradInput.zero_()

        return None,None,gradInput

class roundSTE(torch.autograd.Function):
    """Round {min(Tissue value), max(Tissue value)} a real valued tensor. Backward is STE"""

    @staticmethod
    def forward(ctx, input, dec):
        ctx.input = input
        return torch.round(input, decimals=dec)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None
