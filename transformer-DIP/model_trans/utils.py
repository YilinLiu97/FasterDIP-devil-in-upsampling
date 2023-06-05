import matplotlib.pyplot as plt
import cv2
import os

import torch
import torchvision
from PIL import Image
import numpy as np


class DataLoader:
    def __init__(self, args):
        self.device = args.device
        self.img_list = []
        self.random_list = []
        self.x_list = []
        self.y_list = []
        self.y_dirt_list = []
        self.mask_list = []
        self.num_imgs = 0
        for filename in os.listdir(args.input_dir):
            img = cv2.imread(args.input_dir + filename)
            self.img_list.append(img)
            self.random_list.append(np.random.randn(img.shape[0], img.shape[1], 3))

        self._initialize()

    def _initialize(self):
        for i in range(len(self.random_list)):
            x, y, y_dirt, mask = self._load_data(i)
            self.x_list.append(x)
            self.y_list.append(y)
            self.y_dirt_list.append(y_dirt)
            self.mask_list.append(mask)
        # self.random_list = None
        # self.img_list = None
        self.num_imgs = len(self.x_list)
        print("total number of images:", self.num_imgs)

    def _load_data(self, idx):
        y = self.img_list[idx]

        y_dirt, mask = add_noise(y)

        y = y.transpose((2, 1, 0))
        y_dirt = y_dirt.transpose((2, 1, 0))
        mask = mask.transpose((2, 1, 0))

        y = normalize_img(y)
        y = torch.FloatTensor(y)
        y_dirt = torch.FloatTensor(y_dirt)
        mask = torch.FloatTensor(mask)

        x = torch.zeros([32, y.shape[1], y.shape[2]])
        # x.normal_(mean=0.0, std=0.5)
        x.uniform_()
        x *= 1./5.
        print(x.std())

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        y_dirt = y_dirt.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return x.to(self.device), y.to(self.device), y_dirt.to(self.device), mask.to(self.device)

    def get_data(self, i):
        x, y, y_dirt, mask = self.x_list[i], self.y_list[i], self.y_dirt_list[i], self.mask_list[i]

        return x, y, y_dirt, mask


def add_noise(img):
    img = normalize_img(img)
    noise = np.random.normal(scale=25 / 255., size=img.shape)
    # noise = np.random.uniform(-0.1, 0.1, size=img.shape)
    # scale = 1 / 10.
    # img = np.random.poisson(img * 255 * scale) / scale / 255
    # img = np.clip(img, 0, 1).astype(np.float32)
    img = np.clip(img + noise, 0, 1).astype(np.float32)
    img_dirt = img

    mask = np.ones_like(img_dirt)

    return img_dirt, mask


def normalize_img(img):
    max_v = img.max()
    min_v = img.min()
    if max_v - min_v <= 0:
        return img
    return (img - min_v) / (max_v - min_v)


def store_img(img, img_dir):
    img = img.detach().cpu().numpy()[0]
    img = np.asarray([img[2, :, :], img[1, :, :], img[0, :, :]])
    img = img.transpose((2, 1, 0))
    plots = plt.figure()
    plt.imsave(img_dir, img, format="png")
    plt.close(plots)


def store_fig(psnr_list, ssim_list, loss_list, fig_dir, num):
    iter_nums = np.linspace(1, num, int(num / 100))
    plots = plt.figure(figsize=(15, 4))
    ax = plots.add_subplot(141)
    ax.plot(iter_nums, psnr_list)
    ax.set_ylabel('PSNR')
    ax.set_xlabel('Epochs')

    ax = plots.add_subplot(142)
    ax.plot(iter_nums, ssim_list)
    ax.set_ylabel('SSIM')
    ax.set_xlabel('Epochs')

    ax = plots.add_subplot(143)
    ax.plot(iter_nums, loss_list)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')

    plt.savefig(fig_dir)
    plt.close(plots)


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
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
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
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


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
