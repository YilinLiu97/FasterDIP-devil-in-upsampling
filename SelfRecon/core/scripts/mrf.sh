python mrf.py --task mrf \
                      --model_type DIP_2_scaled \
                      --folder_path '/mnt/yaplab/data/yilinliu/datasets/MRF-DIP/144_8coils' \
                      --gt_mask_path '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/20180206data/180131_1/53' \
                      --save_folder '/mnt/yaplab/data/yilinliu/saves/mrf/subj53' \
                      --batch_size 1 \
                      --progressive False \
                      --num_scales 5 \
                      --need_sigmoid True \
                      --need_tanh False \
                      --need_relu False \
                      --filter_size_down 3 \
                      --filter_size_up 3 \
                      --num_skips 0 \
                      --loss_func l1 \
                      --optimizer adam \
                      --verbose True \
                      --mrf_time_pts 144 \
                      --mrf_interp_mode 'bilinear' \
                      --special 144pts_8coils_invalid999_NoAct \
                      --decay 0.0000001 0.0 \
                      --reg_type 0 \
                      --reg_noise_std 0 \
                      --sr 0.0 \
                      --upsample_mode nearest \
                      --act_func ReLU \
                      --pad zero \
                      --dim 128 \
                      --exp_weight 0 \
                      --iters_print_acc 10 \
                      --num_iters 3000 \
                      --lr 0.05
          
                    
            
