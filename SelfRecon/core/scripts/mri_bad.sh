python mri.py --task mri_knee \
                      --model_type DIP_2_scaled \
                      --folder_path '/shenlab/lab_stor/yilinliu/tmp/' \
                      --save_folder '/mnt/yaplab/data/yilinliu/saves/mri_knee/4X' \
                      --progressive False \
                      --pruning_sensitivity 0.01 \
                      --num_scales 5 \
                      --need_sigmoid False \
                      --filter_size_down 3 \
                      --filter_size_up 3 \
                      --num_skips 5 \
                      --loss_func l1 \
                      --optimizer adam \
                      --jacobian_lbda 0 \
                      --jacobian_eval 0 \
                      --Lipschitz_constant 0 \
                      --Lipschitz_reg 0 \
                      --deepspline_lbda 0.0 \
                      --deepspline_lipschitz False \
                      --verbose True \
                      --ac_factor 6 \
                      --tv_weight 1 \
                      --special 5levels_5skips_256chns_TV_integer1 \
                      --prune_type 'None' \
                      --decay 0.0000001 0.0 \
                      --reg_type 0 \
                      --reg_noise_std 0 \
                      --sr 0.0 \
                      --upsample_mode nearest \
                      --act_func ReLU \
                      --pad zero \
                      --dim 256 \
                      --exp_weight 0 \
                      --num_iters 2500 \
                      --lr 0.008
          
                    
            
