from __future__ import print_function
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import os
import h5py
import time
import csv
import copy
import argparse
from fractions import Fraction

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pytorch_model_summary import summary

from models import getModel
from datasets import getDataset
import configs as cfg
from fit import fit
from mrf_fit import mrf_fit
from losses import getLoss
from visualize import *
from utils import getForwardm
from utils.mri_utils import *
from utils.common_utils import *
from utils.pruning_utils import *
from models.downsampler import Downsampler
from fit import eval_mri, eval_general

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#print("num GPUs", torch.cuda.device_count())
gpu_id = get_vacant_gpu()
torch.cuda.set_device(gpu_id)


def main(args):
    loss_func = getLoss(args.loss_func)
    forwardm = getForwardm(args)
    
    freq_dict = {
        'method': 'log',
        'cosine_only': False,
        'n_freqs': args.n_freqs,
        'base': args.base,
    }
    
    rec2_available = False

    dataset = getDataset(args)
    train_data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=14, pin_memory=True)
    print(f"# of data: {len(train_data)} | batch size: {args.batch_size}")

    assert args.task == 'inpainting'

    for iter, sample in enumerate(train_data):
  
        noisy_target = sample['target_img'].type(dtype)
        orig = sample['gt'].type(dtype)
        mask = sample['mask'].type(dtype)
        filename_full = ' '.join(map(str, sample['filename']))
        filename = filename_full.split("/")[-1]
  
        print('Sample %d %s' % ((iter+1), filename))

        # initialize a net for each input instance
        if not args.progressive:
            args.in_size = orig.shape[-2:]
        
        args.out_size = orig.shape[-2:]
        args.out_chns = orig.shape[1]
  
        args.exp_name = args.save_path + '/' + filename.split(".")[0]
        os.makedirs(args.exp_name, exist_ok=True)

        ### reconstruct ###
        ni_shape = [1] + [args.dim] + list(args.in_size)
        ni, args = get_noise(args.dim, args.noise_method, args.in_size, args.noise_type, freq_dict=freq_dict, args=args)
       
        net = getModel(args).type(dtype)

        if iter == 0:
           print(net)
           with open(args.log_filename, 'at') as log_file:
                log_file.write('---------- Networks initialized -------------\n')
                print(summary(net, torch.zeros(ni.shape).cuda(), show_input=True, show_hierarchical=False))
                log_file.write(summary(net, torch.zeros(ni.shape).cuda(), show_input=True, show_hierarchical=True))
                log_file.write('-----------------------------------------------\n')
        
  
        start = time.time()

        out_avg, trained_net, psnr_score, ssim_score, spectral_norms, singular_list = fit(args,
                                                  net,
                                                  Variable(noisy_target),
                                                  net_input=ni,
                                                  mask=mask,
                                                  downsampler=None,
                                                  orig=orig[0].data.cpu().numpy(),
                                                  apply_f=forwardm,
                                                  experiment_name=args.exp_name,
                                                  snapshots_save_path = args.save_path + '/figures/',
                                                  csm=None,
                                                  unders_recon=None,
                                                  loss_func=loss_func,
                                                  )

        print('\nfinished after %.1f minutes.' % ((time.time() - start) / 60))

        print('PSNR: %.4f  SSIM: %.4f' % (psnr_score, ssim_score))
        rec = trained_net(ni)[0].data.cpu().numpy()

        # smoothing
        if args.exp_weight:
           out_avg = out_avg.data.cpu().numpy()
           rec = out_avg[0] * args.exp_weight + rec * (1 - args.exp_weight)

        metrics = eval_general(torch_to_np(orig), rec)
        baseline_metrics = eval_general(torch_to_np(orig), torch_to_np(noisy_target))
        
        ### pruning
        if args.decay and args.reg_type or args.sr:
            net2 = trained_net  #copy.deepcopy(trained_net)
            print("Pruning...")
            pruning(args, net2, args.pruning_sensitivity)
            rec2 = net2(ni)[0].data.cpu().numpy()
            metrics2 = eval_general(torch_to_np(orig), rec2)
            stat = print_nonzeros(args, net2)
            rec2_available = True


        # Analysis
        info = {'noisy_target': noisy_target[0].data.cpu().numpy(), 'rec': rec, 'orig': orig[0].data.cpu().numpy(), 'p_score': metrics['psnr'], 's_score': metrics['ssim'], 'noisy_p_score': baseline_metrics['psnr'],
   'noisy_s_score': baseline_metrics['ssim'], 'filename': filename_full, 'savename': (f"{args.exp_name}/results")}


        visualize(info)
        save_scores(args, info)

        if rec2_available:
           info2 = {'noisy_target': bicubic_img[0].data.cpu().numpy(), 'rec': rec2, 'orig': orig[0].data.cpu().numpy(), 'p_score': metrics2['psnr'], 's_score': metrics2['ssim'], 'noisy_p_score': noisy_metrics['psnr'],
                'noisy_s_score': noisy_metrics['ssim'], 'filename': filename_full, 'savename': (f"{args.exp_name}/pruning_results")}
           visualize(info2)
           save_scores(args, info2)

        plot_metrics(f"{args.exp_name}/{args.num_iters}th_epoch.pt", f"{args.exp_name}/metric_plots")



if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
   
    def str2float(v):
        return float(sum(Fraction(s) for s in v.split()))
 
    def str2int(v):
        return int(v)

    def str2none(v):
        if v == None or v == 'None' or v == 'none':
           return None
        else:
           v
      #  else:
      #     return int(v)

    parser = argparse.ArgumentParser()   

    parser.add_argument('--exp_name', default='saves/')
    parser.add_argument('--special', default=None, type=str)
    parser.add_argument('--folder_path', default='/shenlab/lab_stor/yilinliu/multicoil_val/')
    parser.add_argument('--gt_mask_path', default='/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/20180206data/180131_1/53', type=str, help='for mrf')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--save_folder', default='/mnt/yaplab/data/yilinliu', type=str)
    parser.add_argument('--log_filename', default='', type=str)
    parser.add_argument('--csv_filename', default='', type=str)
    parser.add_argument('--prepruning', default=None, type=str, help='avg | multi | none')
    parser.add_argument('--verbose', default=False, type=str2bool, help='print losses during training if Yes')

    # task related
    parser.add_argument('--task', default='mri_knee',
                        help='mri_knee | mri_brain | sr | denoising | inpainting')

    # mri data related
    parser.add_argument('--ac_factor', default=4, type=str2int,
                        help='acceleration factor')
    parser.add_argument('--center_frac', default=0.07, type=str2float,
                        help='percentage of the preserved center portion of the k-space')
    parser.add_argument('--mrf_time_pts', default=144, type=str2int)
    parser.add_argument('--mrf_interp_mode', default='bilinear', type=str)
    parser.add_argument('--mrf_padding_mode', default='border', type=str, help='zeros | border | reflection')
    parser.add_argument('--skip_frames', default=1, type=str2int)

    # natural data related
    parser.add_argument('--input_image_name', default='image_F16_512rgb.png', type=str, help='Use image as the input instead of noise')
    parser.add_argument('--noise_sigma', default=25, type=str2int)
    parser.add_argument('--poisson_scale', default=0, type=str2float)
    parser.add_argument('--mask_type', default='Bernoulli', type=str)
    parser.add_argument('--noise_method', default='noise', type=str, help='noise | meshgrid | fourier')
    parser.add_argument('--noise_type', default='u', type=str, help='u | n, uniform or normal')
    parser.add_argument('--cosine_only', default=False, type=str2bool, help='for positional encoding')
    parser.add_argument('--n_freqs', default=8, type=str2int, help='for positional encoding')
    parser.add_argument('--base', default=2 ** (8 / (8 - 1)), type=str2float, help='for positional encoding')
    parser.add_argument('--zero_fraction', default=0.5, type=str2float)
    parser.add_argument('--imsize', default=[-1], nargs='+', type=str2int)
    parser.add_argument('--sr_factor', default=4, type=str2int, help='4 | 8')
    parser.add_argument('--sr_kernel_type', default='lanczos2', type=str, help='see models/downsampler.py')

    # pruning related
    parser.add_argument('--prune_type', default=None, type=str2none, help='HS')
    parser.add_argument('--pruning_sensitivity', default=0.01, type=str2float, help='ConvDecoder:0.03 | DIP:0.01')
    parser.add_argument('--dam_lambda', default=0.0, type=str2float, help='lambda=0.001 for DAM')
    parser.add_argument('--meta_img_idx', default=[0,2,4,6,8], nargs='+', type=int, help='the image idexes used for pre-pruning')

    # model related
    parser.add_argument('--reg_noise_std', default=0, type=str2float, help='add noise at each iteration')
    parser.add_argument('--progressive', default=False, type=str2bool, help='whether the image is gradually upsampled')
    parser.add_argument('--model_type', default='ConvDecoder', type=str)
    parser.add_argument('--patch_size', default=16, type=str2int,
                        help='dividing images into tokens')
    parser.add_argument('--num_layers', default=7, type=str2int,
                        help='default:7 in ConvDecoder')
    parser.add_argument('--out_chns', default=3, type=str2int)
    parser.add_argument('--input_dim', default=32, type=str2int)
    parser.add_argument('--dim', default=256, type=str2int,
                        help='number of channels per layer except for the last one')
    parser.add_argument('--in_size', default=[8, 4], nargs='+', type=int)
    parser.add_argument('--out_size', default=[512, 512], nargs='+', type=int)
    parser.add_argument('--filter_size_up', default=3, type=str2int, help='filter size for the decoder')
    parser.add_argument('--filter_size_down', default=5, type=str2int, help='filter size for the encoder')
    parser.add_argument('--num_skips', default=2, type=str2int, help='number of skip connections')
    parser.add_argument('--norm_func', default='bn', type=str, help='bn | instance')
    parser.add_argument('--need_dropout', default=False, type=str2bool)
    parser.add_argument('--need_sigmoid', default=False, type=str2bool)
    parser.add_argument('--need_tanh', default=False, type=str2bool)
    parser.add_argument('--need_relu', default=False, type=str2bool)
    parser.add_argument('--num_scales', default=5, type=str2int, help='for DIP setup')
    parser.add_argument('--act_func', default='ReLU', type=str, help='ReLU|LeakyReLU|Swish|ELU|GELU')
    parser.add_argument('--upsample_mode', default='nearest', help='nearest|bilinear')
    parser.add_argument('--downsample_mode', default='stride', help='stride|avg|max|lanczos2')
    parser.add_argument('--exit_layer_idx_prior', default=0.2, help='1/p: the expected layer idex to exit')
    parser.add_argument('--min_tau', default=1, type=str2float, help='the temperature for Gumbel-softmax. The smaller the one-hotter.')
    parser.add_argument('--pad', default='zero', help='zero|reflection')

    # optimization related
    parser.add_argument('--optimizer', default='adam', type=str, help='rmsprop | adam | sgd')
    parser.add_argument('--decay_lr', default=False, type=str2bool, help='decay the learning rate, linear | cosine')
    parser.add_argument('--morph_lbda', default=0, type=str2float, help='weight for morph net')
    parser.add_argument('--tv_weight', default=0, type=str2float)
    parser.add_argument('--freq_lbda', default=0, type=str2float, help='weight for the spectral loss')
    parser.add_argument('--jacobian_lbda', default=0, type=str2float, help='weight for jacobian regularizer. default:0.001')
    parser.add_argument('--jacobian_eval', default=False, type=str2bool, help='evaluate the 2-norm of the jacobian matrix')
    parser.add_argument('--Lipschitz_constant', default=0, type=str2float, help='the constant c')
    parser.add_argument('--Lipschitz_reg', default=0, type=str2float, help='whether to learn the lipschitz constant')
    parser.add_argument('--deepspline_lbda', default=0, type=str2float, help='to regulate the strength of deepspline activation functions')
    parser.add_argument('--deepspline_lipschitz', default=False, type=str2bool, help='whether to add regularization loss on deepspline')
    parser.add_argument('--num_power_iterations', default=0, type=str2int)
    parser.add_argument('--batch_size', default=1, type=str2int)
    parser.add_argument('--step_size', default=50, type=str2int, help='the step size for linearly decayed learning rate')
    parser.add_argument('--gamma', default=0.75, type=str2float, help='the decayed rate for linearly decayed learning rate')
    parser.add_argument('--T_max', default=1, type=str2int, help='# of changes')
    parser.add_argument('--loss_func', default='l1', type=str)
    parser.add_argument('--freq_loss_func', default='ehm', type=str)
    parser.add_argument('--every_n_iter', default=200, type=str2int, help='print every n iterations')
    parser.add_argument('--iters_print_acc', default=100, type=str2int)
    parser.add_argument('--num_iters', default=2500, type=str2int)
    parser.add_argument('--reg_type', default=0, type=str2int,
                        help='regularization type: 0:None 1:L1 2:Hoyer 3:HS 4:Transformed L1')
    parser.add_argument('--decay', default=[0.0, 0.0], nargs='+', type=str2float, help='0.0000001 for element-wise HS, 0.00001 for struc-wise')
    parser.add_argument('--exp_weight', default=0.99, type=str2float, help='smoothing weight')
    parser.add_argument('--sr', default=0.0, type=str2float, help='sparsity rate for network slimming')
    parser.add_argument('--lr', type=str2float, default=0.008)
    parser.add_argument('--param_idx', default=[], nargs='+', type=int, help='the layer index for printing out the learning rate, [0,8,16,28]')

    args = parser.parse_args()

    args.save_path = f"{args.save_folder}/{args.task}/{args.model_type}"

    if args.special is not None:
       args.save_path += f"_{args.special}"
        

    print(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
 
    args.log_filename = os.path.join(args.save_path, 'log.txt')
    args.csv_filename = os.path.join(args.save_path, 'results.csv')

    print("------------ Input arguments: ------------")
    for key, val in vars(args).items():
        print(f"{key} {val}")
    print("---------------- End ----------------")

    with open(args.log_filename, 'wt') as log_file:
       log_file.write('------------ Options -------------\n')
       for k, v in sorted(vars(args).items()):
           log_file.write('%s: %s\n' % (str(k), str(v)))
       log_file.write('-------------- End ----------------\n')

    with open(args.csv_filename, 'wt') as csv_file:
       writer = csv.writer(csv_file)
       headers = ['ID', 'PSNR', 'SSIM', 'Depth']
       writer = csv.DictWriter(csv_file, fieldnames=headers)
       writer.writeheader()

    main(args)
