# Faster-DIP-Devil-in-Upsampling

Codebase for our ICCV2023 paper "*The Devil is in the Upsampling: Architectural Decisions Made Simpler for Denoising with Deep Image Prior*". [preprint](https://arxiv.org/pdf/2304.11409.pdf)

We discover that the *unlearnt* upsampling is the main driving force behind the denoising phenomenon (and probably other image restoration tasks as well, e.g., super resolution) when the Deep Image Prior[(DIP)](https://arxiv.org/pdf/1711.10925.pdf) paradigm is used, and translate this finding into practical DIP architectural design for every image without the laborious search

## Dependencies
To install the environment, run:
```shell script
conda env create -f environment.yml 
source activate selfrecon
```
## Dataset
We find that DIP architectural deisgn should be associated with image texture for more effective denoising, and thus build the *Texture-DIP-dataset*, which consists of three popular datasets re-classified into several predifined width choices based on the complexity of image texture. We also include the classic dataset [Set9](https://webpages.tuni.fi/foi/GCF-BM3D/BM3D_TIP_2007.pdf) in `DIP-Recon/data/`, which can be used to replicate the validation experiments presented in our paper.

## Organization
All training scripts for replicating the experiments can be found in `DIP-Recon/scripts/`.

For Figure 3 (a) (the importance of upsampling), run the following respectively:
```shell script
./scripts/denoising_dd_noise.sh
```
```shell script
./scripts/denoising_dd_BilinearUp.sh
```
```shell script
./scripts/denoising_dd_TransUp.sh
```
and similarly for Figure 3 (b), just change the `--model_type` from `DD` to `ConvDecoder`.

For Figure 4 (testing of customized upsamplers), modify and run:
```shell script
./scripts/cd_lpf.sh
```

## Extension to Transformers
For detailed instructions please see `DIP-Recon/transformer-DIP`.

*More to come*.

## Citation
```shell script
@article{liu2023devil,
  title={The Devil is in the Upsampling: Architectural Decisions Made Simpler for Denoising with Deep Image Prior},
  author={Liu, Yilin and Li, Jiang and Pang, Yunkui and Nie, Dong and Yap, Pew-thian},
  journal={arXiv preprint arXiv:2304.11409},
  year={2023}
}
```


