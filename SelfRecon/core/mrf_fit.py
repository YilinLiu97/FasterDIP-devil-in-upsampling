import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
import timm
from timm.loss import JsdCrossEntropy
from utils.mri_utils import *
from utils.mrf_utils import *
from utils.common_utils import *
from utils.pruning_utils import *
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from pytorch_msssim import ms_ssim
from pruning.DAM import *
from pruning.morphnet import *
from losses import getLoss, tv_loss
import dsntnn

import copy
import os


def eval_mrf(gt, pred, s, tmask=None):
   print(f"Retreived tmaps: {np.unique(pred*tmask)}")
   gt, pred = gt / s, pred / s # normalization
   if tmask is not None:
      pred = pred * tmask
      L1re = np.sum((np.abs(pred-gt)/gt)*tmask) / np.sum(tmask) #(np.sum((np.abs(pred-gt)/gt)))*tmask))/np.sum(tmask)
   else:
      L1re = None
   s, p = compare_ssim(gt, pred), compare_psnr(gt, pred) 
   return {'ssim': s, 'psnr': p, 'L1re': L1re}

def mrf_fit(args,
        net,
        img_noisy_var,
        net_input,
        tmask,
        kmask,
        downsampler,
        orig,
        m0map,
        apply_f,
        experiment_name,
        snapshots_save_path,
        csm,
        loss_func = nn.L1Loss(),
        ):

    freq_loss = getLoss(args.freq_loss_func)

    p = [x for x in net.parameters()]

    mse_wrt_noisy = np.zeros(args.num_iters)

    ### optimizer
    print(f"optimize with {args.optimizer}", args.lr)
    
    if args.model_type == 'DIP_ds':
       optimizer = getOptimizer(net.parameters_no_deepspline(), args)
       ds_optimizer = getOptimizer(net.parameters_deepspline(), args)
    else:
       optimizer = getOptimizer(p, args)
       ds_optimizer = None

    if args.decay_lr:  
      print(f"With annealed learning rate ")
      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=args.gamma)

    if len(args.param_idx) != 0:
      rows, cols = (args.num_iters, len(args.param_idx))
      lr_list = torch.zeros(rows, cols) #[[0 for i in range(cols)] for j in range(rows)]

    T1_PSNR_list, T1_SSIM_list, T1_L1re_list, T2_PSNR_list, T2_SSIM_list, T2_L1re_list, Loss_list = [], [], [], [], [], [], []

    os.makedirs(snapshots_save_path, exist_ok=True)

    reg_noise_std = args.reg_noise_std #1./30. # set to 1./20. for sigma=50
    noise = net_input.clone()
    net_input_saved = net_input.clone()
    print("With noise pertubation? ", (reg_noise_std>0))
    out_avg = None

    ### main optimization loop
    dict, r = prepare_dictionary(args.mrf_time_pts)
  
    kld = nn.KLDivLoss()
#    prior_dist = 
    for i in tqdm(range(args.num_iters)):

        optimizer.zero_grad()
        if ds_optimizer is not None:
           ds_optimizer.zero_grad()

        if reg_noise_std > 0:
           net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        net_input = Variable(net_input, requires_grad=True).cuda()
        
        # forward pass
        out = net(net_input)
        print(f"out: {out.shape}")
 #       print("t1 loc min: ", torch.min(out[0,0,...]))
        print("t1 loc mean: ", torch.mean(out[0,0,...]))
 #       print("t2 loc min: ", torch.min(out[0,1,...]))
        print("t2 loc mean: ", torch.mean(out[0,1,...]))
        total_loss = loss_func(apply_f(out, dict, csm, m0map, kmask, tmask), img_noisy_var)
        

        print("meas norm: ", torch.linalg.norm(img_noisy_var))

        if args.verbose:
            print('task loss: ', total_loss)

        '''        
        # smoothing
        if args.exp_weight:
           if out_avg is None:
              out_avg = out.detach()
           else:
              out_avg = out_avg * args.exp_weight + out.detach() * (1 - args.exp_weight)
        else:
           out_avg = out.detach()
       '''

        reg_elem = reg_struc = 0.0
        if args.reg_type and args.decay:
           reg_elem = reg_struc  = 0.0
           for name, param in net.named_parameters():
               if param.requires_grad and torch.sum(torch.abs(param))>0:
                  if args.reg_type == 1:  # L1
                     reg_elem += torch.sum(torch.abs(param))
                  elif args.reg_type == 2:  # Hoyer
                     reg_elem += torch.sum(torch.abs(param)) / torch.sqrt(torch.sum(param ** 2))
                  elif args.reg_type == 3:  # Hoyer Square
                     reg_elem += (torch.sum(torch.abs(param)) ** 2) / torch.sum(param ** 2)
                  elif args.reg_type == 4:  # Transformed L1
                     reg_elem += torch.sum(2 * torch.abs(param) / (1 + torch.abs(param)))           
                  elif args.reg_type == 5: # L2
                     reg_elem += torch.sum(param ** 2)
                  else:
                     reg_elem = 0.0

               if args.decay[1] != 0:
                  if param.requires_grad and 'weight' in name:
                    if args.reg_type == 3:
                        if len(param.shape)==4:
                            reg_struc += ( (torch.sum(torch.sqrt(torch.sum(param**2,(0,2,3))))**2) + (torch.sum(torch.sqrt(torch.sum(param**2,(1,2,3))))**2) )/torch.sum(param**2)
                        elif len(param.shape)==2:
                            reg_struc += ( (torch.sum(torch.sqrt(torch.sum(param**2,0)))**2) + (torch.sum(torch.sqrt(torch.sum(param**2,1)))**2) )/torch.sum(param**2)
                    else:
                        raise NotImplementedError('Regularizer [%s] is not implemented for structural sparsity' % reg_type)

           if args.verbose:
              print("reg_loss: ", args.decay[0] * reg_elem + args.decay[1] * reg_struc)    
           total_loss += args.decay[0] * reg_elem + args.decay[1] * reg_struc 

        if args.tv_weight:
           tv_reg = tv_loss(out, args.tv_weight)
           print("tv reg: ", tv_reg)
           total_loss += tv_reg

        if args.jacobian_lbda:
           grad = torch.autograd.grad(out, net_input, grad_outputs=torch.ones_like(out), retain_graph=True, allow_unused=True)
           if args.verbose:
              print("jacobian reg: ", torch.norm(grad[0], 2) * args.jacobian_lbda)
           total_loss += torch.norm(grad[0], 2) * args.jacobian_lbda

        if args.jacobian_eval:
           grad = torch.autograd.grad(out, net_input, grad_outputs=torch.ones_like(out), retain_graph=True, allow_unused=True)
           if args.verbose:
              print("jacobian eval: ", torch.norm(grad[0], 2))

        if args.Lipschitz_reg:
           lip_loss = torch.tensor(1)
           for name, p in net.named_parameters():
               if 'weight_c' in name:
                  lip_loss = lip_loss * torch.max(p) #nn.functional.softplus(p))
           if args.verbose:
              print("lip_loss: ", lip_loss)
           total_loss += lip_loss

        if args.deepspline_lbda:
           if args.deepspline_lipschitz:
              ds_lip_loss = args.deepspline_lbda * net.BV2()
              total_loss += ds_lip_loss
              if args.verbose:
                 print("ds_lip_loss: ", ds_lip_loss)
           else:
              ds_loss = args.deepspline_lbda * net.TV2()
              total_loss += ds_loss
              if args.verbose:
                 print("ds_loss: ", ds_loss)

        if args.freq_lbda:
           total_loss += args.freq_lbda * freq_loss(out, img_noisy_var)
 
        total_loss.backward()
        optimizer.step()

        if ds_optimizer is not None:
           ds_optimizer.step()


        if i % 1 == 0:
           # Evaluation
           tmaps = retrieve_tmaps(out, r, args.mrf_interp_mode, args.mrf_padding_mode) # (2, 256, 256)
           print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
           print(f"T1gt: {orig[0,...]}")
           print(f"T2gt: {orig[1,...]}")
           metrics_t1 = eval_mrf(orig[0,...], torch_to_np(tmaps[:,0,...]), 5000, torch_to_np(tmask))
           metrics_t2 = eval_mrf(orig[1,...], torch_to_np(tmaps[:,1,...]), 500, torch_to_np(tmask))

           if i % args.iters_print_acc == 0:
              print('Iteration %05d   T1 PSNR_gt: %f  T1 SSIM_gt: %f  T1 L1re: %f  T2 PSNR_gt: %f  T2 SSIM_gt: %f  T2 L1re: %f' % (
            i,   metrics_t1['psnr'], metrics_t1['ssim'], metrics_t1['L1re'], metrics_t2['psnr'], metrics_t2['ssim'], metrics_t2['L1re']))

           T1_PSNR_list.append(metrics_t1['psnr'])
           T1_SSIM_list.append(metrics_t1['ssim'])
           T1_L1re_list.append(metrics_t1['L1re'])
 
           T2_PSNR_list.append(metrics_t2['psnr'])
           T2_SSIM_list.append(metrics_t2['ssim'])
           T2_L1re_list.append(metrics_t2['L1re'])

           Loss_list.append(total_loss.data.cpu())

        if (i + 1) % args.num_iters == 0:
            torch.save({'ni': net_input,
                        'net': net.state_dict(),
                        'T1_PSNR_list': T1_PSNR_list,
                        'T1_SSIM_list': T1_SSIM_list,
                        'T1_L1re_list': T1_L1re_list,
                        'T2_PSNR_list': T2_PSNR_list,
                        'T2_SSIM_list': T2_SSIM_list,
                        'T2_L1re_list': T2_L1re_list,
                        'Loss_list': Loss_list,
                        'out': out.data.cpu().numpy(),
                        'tmaps': tmaps.data.cpu().numpy()
                        }, f"{experiment_name}/{i+1}th_epoch.pt")

        best_net = net

    # get the final tmaps
    loc = net(net_input)
    rec = retrieve_tmaps(loc, r, args.mrf_interp_mode, args.mrf_padding_mode)
    metrics_t1 = eval_mrf(orig[0,...], torch_to_np(rec[:,0,...]), 5000, torch_to_np(tmask))
    metrics_t2 = eval_mrf(orig[1,...], torch_to_np(rec[:,1,...]), 500, torch_to_np(tmask))
    return rec.data.cpu().numpy(), best_net, metrics_t1, metrics_t2




