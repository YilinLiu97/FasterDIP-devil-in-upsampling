1. If merge to selfrecon, the models are in model_trans dir
model_trans/transformer.py: swin U-Net with upsampling
model_trans/transformer2.py: original swin U-Net

How to call the model:
see model_trans/swin.py: (transformer.py and transformer2.py provide the same interface)
self.model = SwinTransformerSys(img_size=img_size, patch_size=patch_size, in_chans=input_chns)
patch_size: needs to be fixed to 4
img_size: an integer number. The image should be a square
in_chans: number of input channels, set 32

Note: the input x to the U-Net
x = torch.zeros([1, 32, img_size, img_size])
x.uniform_()
x *= 1./5.


hyper parameters modified:
reg_noise_std 1./50
exp_weight 0.99
num_iters 3000
lr 0.0008
input_dim 32

python train_unsup.py --task denoising \
                      --model_type set_a_name_here \
                      --folder_path 'tmp' \
                      --save_folder 'tmp' \
                      --progressive False \
                      --pruning_sensitivity 0.01 \
                      --need_sigmoid true \
                      --loss_func l1 \
                      --optimizer adam \
                      --verbose False \
                      --prune_type 'None' \
                      --reg_noise_std 1./50 \
                      --sr 0.0 \
                      --exp_weight 0.99 \
                      --num_iters 3000 \
                      --lr 0.0008 \
                      --input_dim 32 \
                      --tv_weight 0 \
                      --reg_type 0 \
                      --decay 0 0.0 \
                      --special set_a_name_here