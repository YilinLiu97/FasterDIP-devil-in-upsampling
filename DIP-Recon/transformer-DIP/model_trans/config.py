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
    parser.add_argument('--display_epoch', default=100, type=int, help='node rank for distributed training')
    parser.add_argument('--epochs', default=3000, type=int)
    parser.add_argument('--lr1', default=0.0008, type=float, help='learning rate for netwrok')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training')
    parser.add_argument('--device', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--input_dir', default='./data/denoising/', type=str, help='source dir')
    parser.add_argument('--img_dir', default='./img/denoising/', type=str, help='dir to store results')
    parser.add_argument('--use_sigmoid', default='true', type=str2bool)

    opt = parser.parse_args(args=[])

    return opt
