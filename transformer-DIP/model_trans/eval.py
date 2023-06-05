import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_psnr(x, y):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return 100
    max_pixel = max(x.max(), y.max())
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(x, y):
    x = x.detach().cpu().numpy()[0]
    y = y.detach().cpu().numpy()[0]
    x = x.transpose(1, 2, 0)
    y = y.transpose(1, 2, 0)
    ssim_v = ssim(y, x, multichannel=True)
    return ssim_v


class Metric:
    def __init__(self):
        self.psnr_list = []
        self.ssim_list = []
        self.loss_list = []

    def evaluate(self, x, y, loss):
        self.psnr_list.append(compute_psnr(x, y))
        self.ssim_list.append(compute_ssim(x, y))
        self.loss_list.append(loss.item())
