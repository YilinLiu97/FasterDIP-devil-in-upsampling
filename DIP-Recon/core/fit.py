import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
from utils.mri_utils import *
from utils.common_utils import *
from utils.pruning_utils import *
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from pytorch_msssim import ms_ssim
from pruning.DAM import *
from pruning.morphnet import *
from losses import getLoss, tv_loss

import copy
import os


def eval_mri(gt, pred):
   gt, pred = normalize(gt, pred)
   s, p = ssim(np.array([gt]), np.array([pred])), psnr(np.array([gt]), np.array([pred])) #, ms_ssim(gt, pred,data_range=pred.max()).data.cpu().numpy()[np.newaxis][0]
   return {'ssim': s, 'psnr': p}

def eval_general(gt, pred):
   s = compare_ssim(gt.transpose(1,2,0), pred.transpose(1,2,0), multichannel=True)
   p = compare_psnr(gt.transpose(1,2,0), pred.transpose(1,2,0))
#   ms = ms_ssim(gt, pred,data_range=pred.max()).data.cpu().numpy()[np.newaxis][0]
   return {'ssim': s, 'psnr': p}


def updateBN(net, sr):
   for m in net.modules():
       if isinstance(m, nn.BatchNorm2d):
          m.weight.grad.data.add_(sr*torch.sign(m.weight.data)) # L1

def sparsity_penalty(net):
    penalty = []
    for m in net.modules():
        if isinstance(m, DAM_2d):
            penalty.append(torch.clamp(m.beta, min=-5.0)) 
    penalty = torch.cat(penalty)
    return penalty

def get_current_lr(optimizer, parameter_idx, group_idx=0):
    # Adam has different learning rates for each paramter. So we need to pick the
    # group and paramter first.
    group = optimizer.param_groups[group_idx]
    p = group['params'][parameter_idx]

    beta1, _ = group['betas']
    state = optimizer.state[p]

    bias_correction1 = 1 - beta1 ** state['step']
    current_lr = group['lr'] / bias_correction1 / torch.sqrt(state['exp_avg_sq'] + 1e-8)
    return current_lr
         
def fit(args,
        net,
        img_noisy_var,
        net_input,
        mask,
        downsampler,
        orig,
        apply_f,
        experiment_name,
        snapshots_save_path,
        csm = None,
        unders_recon = None,
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

    PSNR_list, SSIM_list, Loss_list, Noisy_PSNR_list, Noisy_SSIM_list = [], [], [], [], []

    os.makedirs(snapshots_save_path, exist_ok=True)

    reg_noise_std = args.reg_noise_std #1./30. # set to 1./20. for sigma=50
    noise = net_input.clone()
    net_input_saved = net_input.clone()
    print("With noise pertubation? ", (reg_noise_std>0))
    out_avg = None

    ### main optimization loop
    nonzero_list = []
    norms_list = []
    singular_list = []
    if args.num_power_iterations: 
       norms, w_svd = spectral_norm(net, args.num_power_iterations)
       norms_list.append(norms)
       singular_list.append(w_svd)

    for i in tqdm(range(args.num_iters)):

        if (i+1) % 10 == 0:
           nonzero_ratio = plot_layer_importance(net, f"{snapshots_save_path}/layer_importance_iter{i+1}.png", args.pruning_sensitivity)
           nonzero_list.append(nonzero_ratio)

        optimizer.zero_grad()
        if ds_optimizer is not None:
           ds_optimizer.zero_grad()

        if reg_noise_std > 0:
           net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        net_input = Variable(net_input, requires_grad=True).cuda()
        
        # forward pass
        out = net(net_input)
       
        if args.task == 'mri_knee' or args.task == 'mrf':
           total_loss = loss_func(apply_f(out, mask), img_noisy_var)
        else:
           total_loss = loss_func(apply_f(out, mask, downsampler), img_noisy_var)

        if args.verbose:
           print('task loss: ', total_loss)

        # smoothing
        if args.exp_weight:
           if out_avg is None:
              out_avg = out.detach()
           else:
              out_avg = out_avg * args.exp_weight + out.detach() * (1 - args.exp_weight)
        else:
           out_avg = out.detach()
        
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

        if args.sr:
          updateBN(net, args.sr)

        optimizer.step()
        if ds_optimizer is not None:
           ds_optimizer.step()

        if args.decay_lr:
           lr_scheduler.step()

        if args.optimizer == 'adam' and len(args.param_idx) != 0:
           for k in range(len(args.param_idx)):
               idx = args.param_idx[k]
               curr_lr =  get_current_lr(optimizer, idx, group_idx=0)
               lr_list[i,k] = torch.mean(curr_lr)

        # deplay the fitting loss every 10 iterations
        if i % 1 == 0:
            # Evaluation
            if args.task == 'mri_knee' or args.task == 'mri_brain':
                img_out = out.squeeze(0)
                ncoils = img_out.shape[0] // 2
                img_out = torch.stack((img_out[:ncoils, :, :], img_out[ncoils:,:,:]), -1)
                imgout = combine_coil_rss(img_out, orig.shape)
                metrics = eval_mri(orig, imgout)
                metrics_baseline = eval_mri(orig, unders_recon)
                
                print('Iteration %05d  baseline_psnr: %f  baseline_ssim: %f  PSRN_gt: %f SSIM_gt: %f' % (
            i,  metrics_baseline['psnr'], metrics_baseline['ssim'], metrics['psnr'], metrics['ssim']))
            else:
                metrics = eval_general(orig, torch_to_np(out_avg))

                if i % args.iters_print_acc == 0:
                   if args.task != 'sr':
                      metrics_noisy = eval_general(torch_to_np(img_noisy_var), torch_to_np(out_avg))
                      psnr_noisy = metrics_noisy['psnr']
                      metrics_noisy_baseline = eval_general(orig, torch_to_np(img_noisy_var))
                      noisy_baseline = metrics_noisy_baseline['psnr']
                   else:
                      psnr_noisy = 0.0                   
                      noisy_baseline = 0.0

                   metrics_plain = eval_general(orig, torch_to_np(out))
                   print('Iteration %05d   PSNR_noisy_baseline: %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
            i,  noisy_baseline, psnr_noisy, metrics_plain['psnr'], metrics['psnr']))
 
            PSNR_list.append(metrics['psnr'])
            SSIM_list.append(metrics['ssim'])
            Noisy_PSNR_list.append(metrics_noisy_baseline['psnr'])
            Noisy_SSIM_list.append(metrics_noisy_baseline['ssim'])
            Loss_list.append(total_loss.data.cpu())

        # compute spectral norm for all layers every 10 iterations
        if i % 10 == 0 and args.num_power_iterations: 
            print("spectral norms computed.")
            wsn, w_svd = spectral_norm(net, args.num_power_iterations)
            norms_list.append(wsn)
            singular_list.append(w_svd)

        if (i + 1) % args.num_iters == 0:
            torch.save({'ni': net_input,
                        'net': net.state_dict(),
                        'PSNR_list': PSNR_list,
                        'SSIM_list': SSIM_list,
                        'Loss_list': Loss_list,
                        'out_avg': out_avg.data.cpu().numpy()
                        }, f"{experiment_name}/{i+1}th_epoch.pt")

        best_net = net

    return out_avg, best_net, metrics['psnr'], metrics['ssim'], norms_list, singular_list, metrics_noisy_baseline['psnr'], metrics_noisy_baseline['ssim']



