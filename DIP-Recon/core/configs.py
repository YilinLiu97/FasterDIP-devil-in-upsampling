import torch.nn as nn
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', default='checkpoints/')
    parser.add_argument('--folder_path', default='/shenlab/lab_stor/yilinliu/multicoil_val/')

    # task related
    parser.add_argument('--task', default='mri_knee',
                        help='mri_knee | mri_brain | sr | denoising | inpainting')

    # mri data related
    parser.add_argument('--ac_factor', default=4, type=int,
                        help='acceleration factor')
    parser.add_argument('--center_frac', default=0.07, type=float,
                        help='percentage of the preserved center portion of the k-space')

    # model related
    parser.add_argument('--model_type', default='DIP', type=str)
    parser.add_argument('--patch_size', default=16, type=int,
                        help='dividing images into tokens')
    parser.add_argument('--num_layers', default=7, type=int,
                        help='default:7 in ConvDecoder')
    parser.add_argument('--out_chns', default=3, type=int)
    parser.add_argument('--dim', default=256, type=int,
                        help='number of channels per layer except for the last one')
    parser.add_argument('--in_size', default=[8,4], nargs='+', type=int)
    parser.add_argument('--out_size', default=[512,512], nargs='+', type=int)
    parser.add_argument('--need_dropout', default=False, type=str2bool)
    parser.add_argument('--need_sigmoid', default=False, type=str2bool)
    parser.add_arugment('--num_scales', default=5, type=int, help='for DIP setup')
    parser.add_argument('--act_func', default='LeakyReLU', type=str)
    parser.add_argument('--upsample_mode', default='nearest', help='nearest|bilinear')

    # optimization related
    parser.add_argument('--loss_func', default='l1', type=str)
    parser.add_argument('--every_n_iter', default=1, type=int, help='print every n iterations')
    parser.add_argument('--num_iters', default=2500, type=int)
    parser.add_argument('--reg_type', default=3, type=int,
                        help='regularization type: 0:None 1:L1 2:Hoyer 3:HS 4:Transformed L1')
    parser.add_argument('--decay', default=0.0000001, type=float)
    parser.add_argument('--lr', type=float, default=0.008)

   

    args = parser.parse_args()

    return args
