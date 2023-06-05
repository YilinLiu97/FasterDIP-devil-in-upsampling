python train_unsup.py --task mri_knee \
                      --model_type DIP_2_scaled \
                      --folder_path '../../multicoil_val/' \
                      --save_folder '/mnt/yaplab/data/yilinliu/saves/mri_knee/test_multicoil_val' \
                      --progressive False \
                      --pruning_sensitivity 0.01 \
                      --num_scales 2 \
                      --need_sigmoid False \
                      --filter_size_down 5 \
                      --filter_size_up 3 \
                      --num_skips 2 \
                      --num_layers 5 \
                      --loss_func l1 \
                      --optimizer adam \
                      --jacobian_lbda 0 \
                      --jacobian_eval 0 \
                      --Lipschitz_constant 0 \
                      --Lipschitz_reg 0 \
                      --deepspline_lbda 0.0 \
                      --deepspline_lipschitz False \
                      --verbose True \
                      --num_power_iterations 10 \
                      --special 2levels_2skips_256chns_3up5down\
                      --prune_type 'None' \
                      --decay 0.0000001 0.0 \
                      --reg_type 0 \
                      --reg_noise_std 0 \
                      --sr 0.0 \
                      --upsample_mode nearest \
                      --need_dropout False \
                      --act_func ReLU \
                      --pad zero \
                      --dim 256 \
                      --exp_weight 0 \
                      --num_iters 3000 \
                      --lr 0.008
          
                    
            
