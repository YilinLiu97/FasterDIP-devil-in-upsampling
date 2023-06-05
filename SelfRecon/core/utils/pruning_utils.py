import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

def pruning(args, model, sensitivity=0.10, ckp=None):
    if ckp is not None:
        checkpoint = torch.load(ckp)
        model = checkpoint['net']
    orig_stdout = sys.stdout
    with open(args.log_filename, 'at') as log_file:
        sys.stdout = log_file
        print("--- Pruning ---")     
        with torch.no_grad():
           for name, p in model.named_parameters():
               if 'mask' in name:
                  continue
               tensor = p.data.cpu().numpy()
               threshold = sensitivity * np.std(tensor)
               print(f'Pruning with threshold : {threshold} for layer {name}')
               new_mask = np.where(abs(tensor) < threshold, 0, tensor)
               p.data = torch.from_numpy(new_mask).cuda()
    sys.stdout = orig_stdout
    return model

def print_nonzeros(args, model, ckp=None):
    if ckp is not None:
        checkpoint = torch.load(ckp)
        model = checkpoint['net']
    nonzero = total = 0
    iter = 0
    col = []
    orig_stdout = sys.stdout
    with open(args.log_filename, 'at') as log_file:
        sys.stdout = log_file
        for name, p in model.named_parameters():
            if 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')

            if 'weight' in name and tensor.ndim == 4:
                iter = iter + 1

            # if iter == 1:
            #    sns.histplot(tensor[0][:50].ravel(), bins=32,stat='count')
            #   col = tensor[0].ravel()
                tensor = np.abs(tensor)
                dim0 = np.sum(np.sum(tensor, axis=0), axis=(1, 2))
                dim1 = np.sum(np.sum(tensor, axis=1), axis=(1, 2))

                nz_count0 = np.count_nonzero(dim0)
                nz_count1 = np.count_nonzero(dim1)

                t = nz_count / total_params
                col.append(t)
                print(f'{name:20} | dim0 = {nz_count0:7} / {len(dim0):7} ({100 * nz_count0 / len(dim0):6.2f}%) | dim1 = {nz_count1:7} / {len(dim1):7} ({100 * nz_count1 / len(dim1):6.2f}%)')
          
        print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')
        sys.stdout = orig_stdout
        return col

def plot_layer_importance(model, savepath, sensitivity=0.01):
    """
    :param model:
    :param sensitivity:
    :return:
    """
    col, names, values = [], [], []
    nonzero = total = 0
    for name, p in model.named_parameters():

        tensor = p.data.cpu().numpy()
        threshold = sensitivity * np.std(tensor)
        tensor = np.where(abs(tensor) < threshold, 0, tensor)

        nz_count = np.count_nonzero(tensor)
        params = np.prod(tensor.shape)
        nonzero += nz_count
        total += params
 
        if 'weight' in name and tensor.ndim == 4:
            t = nz_count / params
            col.append(t)
            names.append(name)
            
    plt.scatter(np.linspace(1, len(col), len(col)), col)
    plt.plot(np.linspace(1, len(col), len(col)), col)
    plt.ylabel('Alive weights (%) of each layer')
    plt.xlabel('The nth convolutional layer')
    plt.grid()

    plt.savefig(savepath + '.png')
    plt.close()

    return nonzero / total

#    plt.hist()

    #print(f'dead: {total-nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% zeros)')
    
