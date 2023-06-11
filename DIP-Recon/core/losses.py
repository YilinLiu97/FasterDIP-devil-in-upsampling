import torch
import torch.nn as nn
import torch.autograd.variable as Variable

from utils.common_utils import *

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

def getLoss(loss_func, alpha=1):
    if loss_func == 'l1':
       loss = nn.L1Loss()
    elif loss_func == 'mse':
       loss = nn.MSELoss()
    elif loss_func == 'bce':
       loss = nn.BCELoss()
    elif loss_func == 'ehm': # exact histogram matching
       loss = EHMLoss()
    elif loss_func == 'focal_freq':
       loss = FocalFrequencyLoss(alpha)
    elif loss_func == 'moment_matching':
       loss = Freq_Statistics_Matching()
    else:
       raise NotImplementedError("No such loss type.")
    return loss

'''
def get_distribution_target(mode='gaussian', target_mean=0.3, length=258):
    if mode == 'gaussian':
       from scipy.stats import norm
       data = np.arange(length)
'''       

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return tv_weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

class EHMLoss(nn.Module):
    def __init__(self):
       super(EHMLoss, self).__init__()
       self.mse = nn.MSELoss()

    def forward(self, psdA, psdB):
       transferred_A = self.exact_feature_distribution_matching(psdA, psdB)
       loss = self.mse(psdA, transferred_A)
       return loss

    def exact_feature_distribution_matching(self, psdA, psdB):
       """
       psdA: output psd 1D
       psdB: target psd 1D
       """
       assert (psdA.size() == psdB.size()) ## content and style features should share the same shape
       B, C = psdA.size(0), psdA.size(1)
       _, index_psdA = torch.sort(psdA)  ## sort content feature
       value_psdB, _ = torch.sort(psdB)      ## sort style feature
       inverse_index = index_psdA.argsort(-1)
       transferred_psdA = psdA + value_psdB.gather(-1, inverse_index) - psdA.detach()
       return Variable(transferred_psdA, requires_grad=True)

class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, ave_spectrum=False, log_matrix=False):
       super(FocalFrequencyLoss, self).__init__()
       self.alpha = alpha
       self.ave_spectrum = ave_spectrum
       self.log_matrix = log_matrix

    def tensor2freq(self, x):
        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(x, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(x, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, freq_rec, freq_target, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            # the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (freq_rec - freq_target) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha # RMSE * alpha

            if self.log_matrix:
                matrix_tmp = matrix_tmp / torch.log(matrix_tmp + 1.0)

            matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :,  None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (freq_rec - freq_target) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None):
        """Forward function to calculate focal frequency loss.
                Args:
                    pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
                    target (torch.Tensor): of shape (N, C, H, W). Target tensor.
                    matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                        Default: None (If set to None: calculated online, dynamic).
                """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) 

class Freq_Statistics_Matching(nn.Module):
    def __init__(self):
        super(Freq_Statistics_Matching, self).__init__()
        self.mse = nn.MSELoss()

    def cal_stats(self, psd1d):
        mu = torch.mean(psd1d)
        diffs = psd1d - mu
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return mu, std, skews, kurtoses

    def cal_mag(self, image):
        return fft_mag(image, log=True)

    def forward(self, output, target):
        assert output.size() == target.size()
        mag2d_rec, mag2d_target = self.cal_mag(output), self.cal_mag(target)
        mu_rec, mu_target = mag2d_rec.mean(), mag2d_target.mean()
        std_rec, std_target = mag2d_rec.std(), mag2d_target.std()
#        psd2d_rec, psd1d_rec = get_psd(output, log=True)
#        psd2d_target, psd1d_target = get_psd(target, log=True)
        #output_mu, output_std, output_skew, output_kurt = self.cal_stats(psd1d_rec)
        #target_mu, target_std, target_skew, target_kurt = self.cal_stats(psd1d_target)
#        print(f"mse(mu): {self.mse(output_mu, target_mu)} mse(std): {self.mse(output_std, target_std)} skew: {output_skew**2} kurt: {output_kurt**2}")
#        print(self.mse(psd1d_rec, psd1d_target))
        return self.mse(mu_rec, mu_target) + \
            self.mse(std_rec, std_target) #+ \
            #output_skew ** 2 + \
            #output_kurt ** 2


