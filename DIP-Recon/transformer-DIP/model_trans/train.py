from model_trans.swin import DIPModel
from model_trans.utils import store_img
from model_trans.eval import Metric
import numpy as np
import torch

import os

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model_conv.utils import torch_to_np, plot_image_grid


def evaluate(i, total_loss, y, real, out, out_avg, show_every, img_dir):

    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    if (i + 1) % show_every == 0:
        # psrn_noisy = peak_signal_noise_ratio(y.detach().cpu().numpy()[0], out.detach().cpu().numpy()[0])
        # psrn_gt = peak_signal_noise_ratio(real.detach().cpu().numpy()[0], out.detach().cpu().numpy()[0])
        if out_avg is not None:
            psrn_gt_sm = peak_signal_noise_ratio(real.detach().cpu().numpy()[0], out_avg.numpy()[0])
            ssim = structural_similarity(real.detach().cpu().numpy()[0], out_avg.numpy()[0], channel_axis=0)
        else:
            psrn_gt_sm = 0
            ssim = 0
        print('Iteration {}, Loss {}, ssim: {}, PSNR_gt_sm: {}'.format(
            i, total_loss, ssim, psrn_gt_sm))
        store_img(out_avg, img_dir + str(i + 1) + ".png")


def optimize(x, y, real, model_net, args, optimizer_type, img_store_dir):
    """Runs optimization loop.
    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations
    """

    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(model_net.parameters(), lr=args.lr1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-5)

    for j in range(args.epochs):
        optimizer.zero_grad()
        total_loss = model_net.update_net(x, y, epoch=j)
        evaluate(j, total_loss, y, real, model_net.out, model_net.out_avg, args.display_epoch, img_store_dir)
        optimizer.step()
        # scheduler.step(total_loss)


def _store_metrics(m, store_dir):
    np.savez(store_dir + "metric.npz", psnr=np.asarray(m.psnr_list), ssim=np.asarray(m.ssim_list),
             loss=np.asarray(m.loss_list))


def train(data_loader, args, idx):
    img_store_dir = args.img_dir + "/{}/".format(idx)
    if not os.path.exists(img_store_dir):
        os.makedirs(img_store_dir)

    x, y, y_dirt, _ = data_loader.get_data(idx)
    print(x.shape)
    store_img(y, img_store_dir + "original.png")
    store_img(y_dirt, img_store_dir + "dirt.png")

    model = DIPModel(img_size=x.shape[2], chns=3)
    model.to(args.device)
    print("start training image", idx)

    optimize(x, y_dirt, y, model, args, 'adam', img_store_dir)


