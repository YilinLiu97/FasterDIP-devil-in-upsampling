import model_trans.config as config
from model_trans.utils import DataLoader
from model_trans.train import train


def main():
    args = config.parse_args()
    data_loader = DataLoader(args)
    for i in range(data_loader.num_imgs):
        train(data_loader, args, i)


if __name__ == '__main__':
    main()
