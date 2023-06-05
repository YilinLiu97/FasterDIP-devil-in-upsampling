#!/bin/bash

# Most parameters don't change during upsampler LPF tests, so define a function to reduce the numbers of paramters passed in 
#conv_dec_denoising(upsample_mode, image_folder, reg_noise_std, exp_weight)
function conv_dec_denoising() 
{
    # Positional paramters
    local upsample_mode=$1
    # Positional paramters with default values 
    local image_folder=${2:-}
    # Default: no noise pertubation
    local reg_noise_std=${3:-0}
    # Default: 0.99 smooth factor
    local exp_weight=${4:-0.99}
    
    local results_save_folder="lpf_$upsample_mode"
    
    if [[ $reg_noise_std != 0 ]]
    then
        # Convert reg_noise_std to 3 digits floating number
        f=$(echo "scale=3; $reg_noise_std" | bc)
        results_save_folder+="_np_$f"
    else
        results_save_folder+="_np_0"
    fi
   
    if [ $exp_weight != 0 ]
    then        
        results_save_folder+="_sm_$exp_weight"
    else
        results_save_folder+="_sm_0"
    fi   
    
    echo "The results will be saved in folder:$results_save_folder"
    
    echo "ConvDecoder --upsample_mode $upsample_mode --special $results_save_folder --folder_path $image_folder --reg_noise_std $reg_noise_std  --exp_weight $exp_weight"
    
    python denoising.py --task denoising \
                      --folder_path $image_folder\
                      --save_folder '/mnt/yaplab/data/yilinliu/saves/LPF/64chns' \
                      --model_type ConvDecoder_LPF \
                      --progressive True \
                      --in_size 16 24 \
                      --num_iters 3000 \
                      --norm_func bn \
                      --noise_sigma 25 \
                      --filter_size_down 3 \
                      --filter_size_up 3 \
                      --num_layers 6 \
                      --num_skips 0 \
                      --need_sigmoid True \
                      --loss_func mse \
                      --freq_loss_func focal_freq \
                      --num_scales 2 \
                      --optimizer adam \
                      --decay_lr False \
                      --step_size 50 \
                      --gamma 0.55 \
                      --morph_lbda 1e-5 \
                      --special $results_save_folder\
                      --prune_type None \
                      --reg_type 0 \
                      --decay 0 0 \
                      --freq_lbda 0 \
                      --reg_noise_std $reg_noise_std \
                      --upsample_mode $upsample_mode \
                      --act_func ReLU \
                      --pad zero \
                      --dim 128 \
                      --lr 0.01 \
                      --exp_weight $exp_weight

}

#data_folder = '/mnt/yaplab/data/yilinliu/datasets/BM3D'

#ConvDecoder Upsampler:nearest NoisePertubation:0  ExpWeight:0
#conv_dec_denoising nearest "data/denoising2/"  0  0

#ConvDecoder Upsampler:bilinear NoisePertubation:0  ExpWeight:0
#conv_dec_denoising bilinear "data/denoising2/" 0  0

#ConvDecoder Upsampler:bicubic NoisePertubation:0  ExpWeight:0
#conv_dec_denoising bicubic "data/denoising2/" 0  0

#ConvDecoder Upsampler:nearest NoisePertubation:0  ExpWeight:0.99
#conv_dec_denoising nearest "/mnt/yaplab/data/yilinliu/datasets/BM3D"  0  0.99

#ConvDecoder Upsampler:bilinear NoisePertubation:0  ExpWeight:0.99
#conv_dec_denoising bilinear "/mnt/yaplab/data/yilinliu/datasets/BM3D" 0  0.99

#ConvDecoder Upsampler:bicubic NoisePertubation:0  ExpWeight:0.99
#conv_dec_denoising bicubic "/mnt/yaplab/data/yilinliu/datasets/BM3D" 0  0.99


#ConvDecoder Upsampler:LPF1 NoisePertubation:0  ExpWeight:0
#conv_dec_denoising LPF1 $data_folder 0  0

#ConvDecoder Upsampler:LPF2 NoisePertubation:0  ExpWeight:0
#conv_dec_denoising LPF2 "data/denoising2/" 0  0

#ConvDecoder Upsampler:LPF3 NoisePertubation:0  ExpWeight:0
#conv_dec_denoising LPF3 "data/denoising2/" 0  0

#ConvDecoder Upsampler:LPF4 NoisePertubation:0  ExpWeight:0
#conv_dec_denoising LPF4 "data/denoising2/" 0  0

#ConvDecoder Upsampler:LPF41 NoisePertubation:0  ExpWeight:0
#conv_dec_denoising LPF41 "data/denoising2/" 0  0

#ConvDecoder Upsampler:LPF5 NoisePertubation:0  ExpWeight:0
#conv_dec_denoising LPF5 "data/denoising2/" 0  0

#ConvDecoder Upsampler:LPF6 NoisePertubation:0  ExpWeight:0
#conv_dec_denoising LPF6 "data/denoising2/" 0  0


#ConvDecoder Upsampler:LPF1 NoisePertubation:0  ExpWeight:0.99
conv_dec_denoising LPF1 "/mnt/yaplab/data/yilinliu/datasets/ours_texture_dataset/Texture_Datasets_NoRepeat/64_2" 0  0.99

#ConvDecoder Upsampler:LPF2 NoisePertubation:0  ExpWeight:0.99
conv_dec_denoising LPF2 "/mnt/yaplab/data/yilinliu/datasets/ours_texture_dataset/Texture_Datasets_NoRepeat/64_2" 0  0.99

#ConvDecoder Upsampler:LPF3 NoisePertubation:0  ExpWeight:0.99
conv_dec_denoising LPF3 "/mnt/yaplab/data/yilinliu/datasets/ours_texture_dataset/Texture_Datasets_NoRepeat/64_2" 0  0.99

#ConvDecoder Upsampler:LPF4 NoisePertubation:0  ExpWeight:0.99
#conv_dec_denoising LPF4 "/mnt/yaplab/data/yilinliu/datasets/ours_texture_dataset/Texture_Datasets_NoRepeat/128" 0  0.99

#ConvDecoder Upsampler:LPF41 NoisePertubation:0  ExpWeight:0.99
#conv_dec_denoising LPF41 "/mnt/yaplab/data/yilinliu/datasets/ours_texture_dataset/Texture_Datasets_NoRepeat/128" 0  0.99

#ConvDecoder Upsampler:LPF5 NoisePertubation:0  ExpWeight:0.99
conv_dec_denoising LPF5 "/mnt/yaplab/data/yilinliu/datasets/ours_texture_dataset/Texture_Datasets_NoRepeat/64_2" 0  0.99

#ConvDecoder Upsampler:LPF6 NoisePertubation:0  ExpWeight:0.99
conv_dec_denoising LPF6 "/mnt/yaplab/data/yilinliu/datasets/ours_texture_dataset/Texture_Datasets_NoRepeat/64_2" 0  0.99
