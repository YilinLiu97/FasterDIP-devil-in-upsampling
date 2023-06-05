import numpy as np

import torch
import torch.nn as nn

def get_uniform_ratio(current, target):
    return current/target

def get_cbns(model):
    convs = []
    bns = []
    for m in model.modules():
        # store the information for batchnorm
        if isinstance(m, nn.Conv2d):
            convs.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bns.append(m)
    return convs, bns

def morph_regularizer(model, constraint='size', cbns=None, maps=None):
    # build kv map
    if cbns is None:
        cbns = get_cbns(model)

    G = torch.zeros([1], requires_grad=True).cuda()
    for idx, (conv, bn) in enumerate(zip(*cbns)):
        if idx < len(cbns[0])-1:
            gamma_prev = torch.abs(bn.weight)
            A = (gamma_prev > 0)
            gamma_now = torch.abs(cbns[1][idx+1].weight)
            B = (gamma_now > 0)
            if constraint == 'size':
                cost = cbns[0][idx+1].weight.size(2)*cbns[0][idx+1].weight.size(3)
            elif constraint == 'flops':
                assert maps is not None, 'Output Map is None!'
                cost = 2 * maps[idx+1][0] * maps[idx+1][0] * cbns[0][idx+1].weight.size(2) * cbns[0][idx+1].weight.size(3)
            G = G + cost * (gamma_prev.sum()*B.sum().type_as(gamma_prev) + gamma_now.sum()*A.sum().type_as(gamma_now))
    return G


def uniform_grow(args, model, growth_rate):
    assert growth_rate > 1

    count = 0
    pre_out_chns = args.dim
    for m in model.modules():

        if isinstance(m, nn.Conv2d):
            count += 1
            conv = m

            # get the number nonzero values
            tensor, b = conv.weight.data.cpu().numpy(), conv.bias.data.cpu().numpy()
            tensor, b = np.abs(tensor), np.abs(b)
            dim0 = np.sum(np.sum(tensor, axis=0), axis=(1, 2))
            dim1 = np.sum(np.sum(tensor, axis=1), axis=(1, 2))

            nz_count0 = np.count_nonzero(dim0)
            nz_count1 = np.count_nonzero(dim1)
            nz_count_b = np.count_nonzero(b)
            print('nz_count0: ', nz_count0)
            print('nz_count1: ', nz_count1)

            #delete all-zero in-channels/out_channels, shape: [nz_count1, nz_count0, x, y]
            tmp0 = tensor[~(tensor == 0).all(axis=(1,2,3))]
            tmp = tmp0.transpose(1,0,2,3)
            tensor = tmp[~(tmp == 0).all(axis=(1,2,3))]
            tensor = tensor.transpose(1,0,2,3)

            assert tensor.shape[0] == nz_count1 and tensor.shape[1] == nz_count0

            in_grown = pre_out_chns if count != 1 else args.dim
            out_grown = int(np.round(nz_count1 * growth_rate)) if conv.out_channels != args.out_chns else args.out_chns
            pre_out_chns = out_grown
            # a mask for bn, marking where gamma and beta should be removed
            out_chns_mask = np.where(dim1==0, 0, 1)

            new_conv = torch.nn.Conv2d(in_channels=in_grown, \
                                        out_channels=out_grown,
                                        kernel_size=conv.kernel_size, \
                                        stride=conv.stride,
                                        padding=conv.padding,
                                        dilation=conv.dilation,
                                        groups=conv.groups,
                                        bias=True)
            conv.in_channels = in_grown
            conv.out_channels = out_grown

            old_weights = tensor
            new_weights = new_conv.weight.data.cpu().numpy()

            old_bias = b[b!=0]
            new_bias = new_conv.bias.data.cpu().numpy()

            new_in_channels = new_weights.shape[1]
            new_out_channels = new_weights.shape[0]
            old_in_channels = old_weights.shape[1] 
            old_out_channels = old_weights.shape[0]

            # initialize the first parts of the new weights as the old weights, and simply set the rest to 0
            if old_in_channels <= new_in_channels and old_out_channels < new_out_channels:
                new_weights[:old_out_channels, :old_in_channels, :,:] = old_weights  # old_weights / np.linalg.norm(old_weights)
                # new_parts = new_weights[old_out_channels:, old_in_channels:, :, :]
                # new_weights[old_out_channels:, old_in_channels:, :, :] = new_parts #new_parts / np.linalg.norm(new_parts)
                new_bias[:nz_count_b] = old_bias
             

            conv.weight.data = torch.from_numpy(new_weights).to('cuda')

            conv.weight.grad = None
            conv.bias.data = torch.from_numpy(new_bias).to('cuda')
            conv.bias.grad = None

        elif isinstance(m, nn.BatchNorm2d):
            next_bn = m
  
            # Surgery on next batchnorm layer
            next_new_bn = \
                torch.nn.BatchNorm2d(num_features=out_grown, \
                                     eps=next_bn.eps, \
                                     momentum=next_bn.momentum, \
                                     affine=next_bn.affine,
                                     track_running_stats=next_bn.track_running_stats)
            next_bn.num_features = out_grown #int(np.round(next_bn.num_features * growth_rate))

            old_weights = next_bn.weight.data.cpu().numpy() * out_chns_mask
            old_weights = old_weights[old_weights != 0]
            new_weights = next_new_bn.weight.data.cpu().numpy()
        

            old_bias = next_bn.bias.data.cpu().numpy() * out_chns_mask
            old_bias = old_bias[old_bias != 0]
            new_bias = next_new_bn.bias.data.cpu().numpy()

            old_running_mean = next_bn.running_mean.data.cpu().numpy() * out_chns_mask
            old_running_mean = old_running_mean[old_running_mean != 0]
            new_running_mean = next_new_bn.running_mean.data.cpu().numpy()
     
            old_running_var = next_bn.running_var.data.cpu().numpy() * out_chns_mask
            old_running_var = old_running_var[old_running_var != 0]
            new_running_var = next_new_bn.running_var.data.cpu().numpy()

            new_weights[:(len(old_weights))] = old_weights
     
            next_bn.weight.data = torch.from_numpy(new_weights).to('cuda')
            next_bn.weight.grad = None

            new_bias[:(len(old_bias))] = old_bias
            next_bn.bias.data = torch.from_numpy(new_bias).to('cuda')
            next_bn.bias.grad = None

            new_running_mean[: (len(old_running_mean))] = old_running_mean
            next_bn.running_mean.data = torch.from_numpy(new_running_mean).to('cuda')
            next_bn.running_mean.grad = None

            new_running_var[: (len(old_running_var))] = old_running_var
            next_bn.running_var.data = torch.from_numpy(new_running_var).to('cuda')
            next_bn.running_var.grad = None

    print(model)


def uniform_grow2(args, model, growth_rate):
    assert growth_rate > 1

    count = 0

    for m in model.modules():
       
        if isinstance(m, nn.Conv2d):
            count += 1
            conv = m
            if conv.groups == conv.out_channels and conv.groups == conv.in_channels:
                new_conv = torch.nn.Conv2d(in_channels=int(np.round(conv.out_channels * growth_rate)), \
                                            out_channels=int(np.round(conv.out_channels * growth_rate)),
                                            kernel_size=conv.kernel_size, \
                                            stride=conv.stride,
                                            padding=conv.padding,
                                            dilation=conv.dilation,
                                            groups=int(np.round(conv.out_channels * growth_rate)),
                                            bias=conv.bias)
                conv.in_channels = int(np.round(conv.out_channels * growth_rate))
                conv.groups = int(np.round(conv.out_channels * growth_rate))
                conv.out_channels = int(np.round(conv.out_channels * growth_rate))
            else:
               
                in_grown = int(np.round(conv.in_channels * growth_rate)) if count != 1 else args.dim
                out_grown = int(np.round(conv.out_channels * growth_rate)) if conv.out_channels != args.out_chns else args.out_chns
                new_conv = torch.nn.Conv2d(in_channels=in_grown, \
                                            out_channels=out_grown,
                                            kernel_size=conv.kernel_size, \
                                            stride=conv.stride,
                                            padding=conv.padding,
                                            dilation=conv.dilation,
                                            groups=conv.groups,
                                            bias=True)
                conv.in_channels = in_grown
                conv.out_channels = out_grown

            old_weights = conv.weight.data.cpu().numpy()
            new_weights = new_conv.weight.data.cpu().numpy()
            old_bias = conv.bias.data.cpu().numpy()
            new_bias = new_conv.bias.data.cpu().numpy()

            new_out_channels = new_weights.shape[0]
            new_in_channels = new_weights.shape[1]
            old_out_channels = old_weights.shape[0]
            old_in_channels = old_weights.shape[1]

            # initialize the first parts of the new weights as the old weights, and simply set the rest to 0
            if old_out_channels < new_out_channels and old_in_channels < new_in_channels:
#                print("Here !!!!!!!!!")
                new_weights[:old_out_channels, :old_in_channels, :, :] = old_weights #old_weights / np.linalg.norm(old_weights)
                #new_parts = new_weights[old_out_channels:, old_in_channels:, :, :]
                #new_weights[old_out_channels:, old_in_channels:, :, :] = new_parts #new_parts / np.linalg.norm(new_parts)
                new_bias[:old_out_channels] = old_bias
      
            elif old_out_channels < new_out_channels:
 #               print("Nope!!!!!!!!!!!!!!!!!!!")
                new_weights[:old_out_channels, :, :, :] = old_weights
                new_bias[:old_out_channels] = old_bias
             

            conv.weight.data = torch.from_numpy(new_weights).to('cuda')
            
#            conv.weight.grad = None
            conv.bias.data = torch.from_numpy(new_bias).to('cuda')
 #           conv.bias.grad = None

        elif isinstance(m, nn.BatchNorm2d):
            next_bn = m
            # Surgery on next batchnorm layer
            next_new_bn = \
                torch.nn.BatchNorm2d(num_features=int(np.round(next_bn.num_features * growth_rate)), \
                                        eps=next_bn.eps, \
                                        momentum=next_bn.momentum, \
                                        affine=next_bn.affine,
                                        track_running_stats=next_bn.track_running_stats)
            next_bn.num_features = int(np.round(next_bn.num_features * growth_rate))

            old_weights = next_bn.weight.data.cpu().numpy()
            new_weights = next_new_bn.weight.data.cpu().numpy()
            old_bias = next_bn.bias.data.cpu().numpy()
            new_bias = next_new_bn.bias.data.cpu().numpy()
            old_running_mean = next_bn.running_mean.data.cpu().numpy()
            new_running_mean = next_new_bn.running_mean.data.cpu().numpy()
            old_running_var = next_bn.running_var.data.cpu().numpy()
            new_running_var = next_new_bn.running_var.data.cpu().numpy()

            new_weights[: old_weights.shape[0]] = old_weights
            new_weights[old_weights.shape[0]:] = 1
            next_bn.weight.data = torch.from_numpy(new_weights).to('cuda')
  #          next_bn.weight.grad = None

            new_bias[: old_bias.shape[0]] = old_bias
            new_bias[old_bias.shape[0]:] = 0
            next_bn.bias.data = torch.from_numpy(new_bias).to('cuda')
   #         next_bn.bias.grad = None

            new_running_mean[: old_running_mean.shape[0]] = old_running_mean
            new_running_mean[old_running_mean.shape[0]:] = 0
            next_bn.running_mean.data = torch.from_numpy(new_running_mean).to('cuda')

            new_running_var[: old_running_var.shape[0]] = old_running_var
            new_running_var[old_running_var.shape[0]:] = 1
            next_bn.running_var.data = torch.from_numpy(new_running_var).to('cuda')


#    print(model)
