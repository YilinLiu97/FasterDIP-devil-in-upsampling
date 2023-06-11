import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import csv
import scipy.io as sio
from collections import OrderedDict
from utils.common_utils import *

def visualize(info):
   orig = info['orig']
   rec = info['rec']
   noisy_target = info['noisy_target']

   if orig.shape[0] == 3 and rec.shape[0] == 3:
      orig = orig.transpose(1,2,0)
      rec = rec.transpose(1,2,0)
      noisy_target = noisy_target.transpose(1,2,0)
   elif orig.shape[0] == 1 and rec.shape[0] == 1: # gray-scale image
      orig, rec, noisy_target = orig[0], rec[0], noisy_target[0]

   plt.imsave(f"{info['savename']}_orig.png", orig)

   plt.imsave(f"{info['savename']}_noisy.png", noisy_target)

   plt.imsave(f"{info['savename']}_recon.png", rec)
   plt.close()

def plot_metrics(ckp_path, savepath):
   ckp = torch.load(ckp_path)
   psnr = ckp['PSNR_list']
   ssim = ckp['SSIM_list']
   loss = ckp['Loss_list']

   bins = len(psnr)
   iter_nums = np.linspace(1, bins, bins)

   plots = plt.figure(figsize=(15,4))
   ax = plots.add_subplot(131)
   ax.plot(iter_nums, psnr)
   ax.set_ylabel('PSNR')
   ax.set_xlabel('Epochs')

   ax = plots.add_subplot(132)
   ax.plot(iter_nums, ssim)
   ax.set_ylabel('SSIM')
   ax.set_xlabel('Epochs')

   ax = plots.add_subplot(133)
   ax.plot(iter_nums, loss)
   ax.set_ylabel('Loss')
   ax.set_xlabel('Epochs')

   plt.savefig(savepath + '.png')
   plt.close()
   
def visualize_mrf(info):
   orig = info['orig'] # (2, 256, 256)
   rec = info['rec'] # (2, 256, 256)
   mask = info['mask'] # (256, 256)
   metrics_t1 = info['t1_metrics']
   metrics_t2 = info['t2_metrics']

   if orig.shape[0] == 1 and rec.shape[0] == 1: # gray-scale image
      orig, rec = orig[0], rec[0]
      
   t1gt, t2gt = orig[0], orig[1]
   t1pred, t2pred = rec[0,0,...], rec[0,1,...]
   
   # Save visual results to mats
   vis = OrderedDict([('pred', rec),('gt', orig), ('mask', mask)])
   sio.savemat(info['savename']+'_tmaps.mat', {'visual_results': vis})

   fig = plt.figure(figsize=(12,5))
   ax = fig.add_subplot(141)
   ax.imshow(t1gt,'gray')
   ax.set_title(info['filename'] + '_T1_Ground_Truth')
   ax.axis('off')

   ax = fig.add_subplot(142)
   ax.imshow(t1pred,'gray')
   ax.set_title('PSNR: %.4f  SSIM: %.4f  L1re: %.4f'%(metrics_t1['psnr'], metrics_t1['ssim'], metrics_t1['L1re']))
   ax.axis('off')
   
   ax = fig.add_subplot(143)
   ax.imshow(t2gt,'gray')
   ax.set_title(info['filename'] + '_T2_Ground_Truth')
   ax.axis('off')
   
   ax = fig.add_subplot(144)
   ax.imshow(t2pred,'gray')
   ax.set_title('PSNR: %.4f  SSIM: %.4f  L1re: %.4f'%(metrics_t2['psnr'], metrics_t2['ssim'], metrics_t2['L1re']))
   ax.axis('off')
   
   plt.savefig(info['savename'] + '.png')
   plt.close()
  # plt.show()
  
def plot_metrics_mrf(ckp_path, savepath):
   ckp = torch.load(ckp_path)
   t1_psnr = ckp['T1_PSNR_list']
   t1_ssim = ckp['T1_SSIM_list']
   t1_L1re = ckp['T1_L1re_list']

   t2_psnr = ckp['T2_PSNR_list']
   t2_ssim = ckp['T2_SSIM_list']
   t2_L1re = ckp['T2_L1re_list']

   loss = ckp['Loss_list']

   bins = len(t1_psnr)
   iter_nums = np.linspace(1, bins, bins)

   plots = plt.figure(figsize=(15,4))
   ax = plots.add_subplot(141)
   ax.plot(iter_nums, t1_psnr)
   ax.set_ylabel('T1 PSNR')
   ax.set_xlabel('Epochs')

   ax = plots.add_subplot(142)
   ax.plot(iter_nums, t1_ssim)
   ax.set_ylabel('T1 SSIM')
   ax.set_xlabel('Epochs')

   ax = plots.add_subplot(143)
   ax.plot(iter_nums, t1_L1re)
   ax.set_ylabel('T1 Relative L1 Error')
   ax.set_xlabel('Epochs')
   
   ax = plots.add_subplot(144)
   ax.plot(iter_nums, loss)
   ax.set_ylabel('Loss')
   ax.set_xlabel('Epochs')

   plt.savefig(savepath + '_T1.png')
   plt.close()

   plots2 = plt.figure(figsize=(15,4))
   ax = plots2.add_subplot(141)
   ax.plot(iter_nums, t2_psnr)
   ax.set_ylabel('T2 PSNR')
   ax.set_xlabel('Epochs')

   ax = plots.add_subplot(142)
   ax.plot(iter_nums, t2_ssim)
   ax.set_ylabel('T2 SSIM')
   ax.set_xlabel('Epochs')

   ax = plots.add_subplot(143)
   ax.plot(iter_nums, t2_L1re)
   ax.set_ylabel('T2 Relative L1 Error')
   ax.set_xlabel('Epochs')
   
   ax = plots.add_subplot(144)
   ax.plot(iter_nums, loss)
   ax.set_ylabel('Loss')
   ax.set_xlabel('Epochs')

   plt.savefig(savepath + '_T2.png')
   plt.close()


def save_scores(args, info):
   filename = info['filename']
   pscore = info['p_score']
   sscore = info['s_score']
   baseline = info['baseline'] if 'baseline' in info.keys() else None

   orig_stdout = sys.stdout
   with open(args.log_filename, 'at') as log_file:
      sys.stdout = log_file
      print(f'--- {filename} ---')
      print(f'{pscore}, {sscore}, {baseline}')
   sys.stdout = orig_stdout
   
   # write to csv file
   csvfile = open(args.csv_filename, 'at')
   writer = csv.writer(csvfile)
   writer.writerow([filename, pscore, sscore, baseline])
   csvfile.close()

def plot_multiple_spectral_norms(norms, savepath):
    norms = np.array(norms).transpose(1,0)
    print("norms in visualize: ", norms.shape)
    means = norms #.mean(0)
    #stds = norms.std(0)
    #print(means)

    iter_nums = np.arange(len(norms.transpose(1,0)))*10
    plt.xlabel("Training Iteration")
    plt.ylabel("Spectral Norm of Layer Weights")
    for layer_num, mean_curve in enumerate(means): 
        p = plt.plot(iter_nums, mean_curve, label=f'Layer {layer_num + 1}')
#        plt.fill_between(iter_nums, mean_curve + std_curve, mean_curve - std_curve, color=p[0].get_color(), alpha=0.15)
    plt.legend()
    plt.savefig(savepath + '.png')
    plt.close()

    torch.save({'spectral_norms': norms}, f"{savepath}_spectral_norms.pt")

def plot_singular_values_all_iters(singular_list, savepath):
    # Singular_list: (num_iters, num_layers, values)
    # plot the singular values at iter #: 30, 100, 250, 300
    sigmas = np.array(singular_list)
    num_iters, num_layers, num_values = sigmas.shape[0], sigmas.shape[1], sigmas.shape[2]
    iter300_values = sigmas[30]
    iter500_values = sigmas[50]
    iter1000_values = sigmas[100]
    iter1500_values = sigmas[150]
    iter2500_values = sigmas[250]
    iter3000_values = sigmas[300]
  
    indexes = np.arange(num_values)
    log_sigmas = np.log(sigmas)

    iter300_log_values = log_sigmas[30]
    iter500_log_values = log_sigmas[50]
    iter1000_log_values = log_sigmas[100]
    iter1500_log_values = log_sigmas[150]
    iter2500_log_values = log_sigmas[250]
    iter3000_log_values = log_sigmas[300]

    for layer in range(num_layers):
        plt.xlabel("Singular value rank index")
        plt.ylabel("Singular value")

        plt.plot(indexes, iter300_values[layer], label=f'0.3k')
        plt.plot(indexes, iter500_values[layer], label=f'0.5k')
        plt.plot(indexes, iter1000_values[layer], label=f'1k')
        plt.plot(indexes, iter1500_values[layer], label=f'1.5k')
        plt.plot(indexes, iter2500_values[layer], label=f'2.5k')
        plt.plot(indexes, iter3000_values[layer], label=f'3k')

        plt.legend()
        plt.savefig(savepath + f'_Layer{layer+1}.png')
        plt.close()          
    
        plt.xlabel("Singular value rank index")
        plt.ylabel("Log-scale singular value")

        plt.plot(indexes, iter300_log_values[layer], label=f'0.3k')
        plt.plot(indexes, iter500_log_values[layer], label=f'0.5k')
        plt.plot(indexes, iter1000_log_values[layer], label=f'1k')
        plt.plot(indexes, iter1500_log_values[layer], label=f'1.5k')
        plt.plot(indexes, iter2500_log_values[layer], label=f'2.5k')
        plt.plot(indexes, iter3000_log_values[layer], label=f'3k')

        plt.legend()
        plt.savefig(savepath + f'log_scale_Layer{layer+1}.png')
        plt.close()

    torch.save({'log_sigmas': log_sigmas,
                'sigmas': sigmas}, f"{savepath}_singular_values.pt")
