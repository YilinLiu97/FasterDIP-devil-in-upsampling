# Faster-DIP-Devil-in-Upsampling

Codebase for our paper "*[The Devil is in the Upsampling: Architectural Decisions Made Simpler for Denoising with Deep Image Prior](https://arxiv.org/pdf/2304.11409.pdf)*". *ICCV 2023*

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
All training scripts for replicating the experiments can be found in `DIP-Recon/scripts/`, and similarly for Figure 3 (b), just change the `--model_type` from `DD` to `ConvDecoder`.

For Figure 4 (testing of customized upsamplers), modify and run:
```shell script
./scripts/cd_lpf.sh
```

## Extend to Transformers
For detailed instructions please refer to `DIP-Recon/transformer-DIP`.

*More to come*.

## Citation
```shell script
@InProceedings{Liu_2023_ICCV,
    author    = {Liu, Yilin and Li, Jiang and Pang, Yunkui and Nie, Dong and Yap, Pew-Thian},
    title     = {The Devil is in the Upsampling: Architectural Decisions Made Simpler for Denoising with Deep Image Prior},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {12408-12417}
}
```


