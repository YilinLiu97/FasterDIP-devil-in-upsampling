# SelfRecon

A DIP (Deep Image Prior) based unsupervised reconstruction framework for general inverse problems. 

Currently support:
- MRI image reconstruction
- Denoising
- Inpainting 
- Super Resolution


## Dependencies

To install all dependencies, simply run:

```shell script
conda env create -f environment.yml
```

## Datasets
FastMRI knee datasets are used for MRI image reconstruction, which can be found [here](https://fastmri.med.nyu.edu). For natural image reconstruction, several classic datasets such as CBM3D, Set5 and Set5 have been included in the `data` folder.

More models/datasets to be added soon! Meanwhile, if you have a customized dataset that needs preprocessings different from the ones above:
- Write a new data loader in `datasets/` -- see `datasets/knee_data.py` or `datasets/denoising.py` for references. 
- Modify `datasets/__init__.py` (add a single line)

## Quick Start
See `scripts` for training scripts for different kinds of reconstruction tasks, and `train_unsup.py` for detailed definitions for the hyperparameters. 

Currently supported models (CNNs, Vision Transformers and MLPs) can be found in `models/__init__.py`. You can also write your own model in `models` and then modify the `__init__.py`.

Take MRI reconstruction as an example:
```shell script
python train_unsup.py --task mri_knee \
                      --model_type DIP_2_scaled \
                      --folder_path '/path/to/data' \
                      --save_folder '/path/to/save' \
                      --progressive False \
                      --pruning_sensitivity 0.01 \
                      --num_scales 2 \
                      --need_sigmoid False \
                      --filter_size_down 3 \
                      --filter_size_up 3 \
                      --num_skips 2 \
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
                      --tv_weight 0 \
                      --special HoyerSquareReg\
                      --prune_type 'None' \
                      --decay 0.0000001 0.0 \
                      --reg_type 3 \
                      --need_dropout False \
                      --reg_noise_std 0 \
                      --sr 0.0 \
                      --upsample_mode nearest \
                      --act_func ReLU \
                      --pad zero \
                      --dim 256 \
                      --exp_weight 0 \
                      --num_iters 3000 \
                      --lr 0.008
```
 
Denoising:
```shell script
./scripts/denoising.sh
```
Inpainting:
```shell script
./scripts/inpainting.sh
```

Super Resolution:
```shell script
./scripts/sr.sh
```
Results are saved in `SelfRecon/saves`.


