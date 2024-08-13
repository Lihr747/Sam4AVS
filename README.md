# SAM4AVS: SAM for zero-shot Audio-Visual Segmentation.
## Introduction
This repository contains the code for the paper "[How Can Contrastive Pre-training Benefit Audio-Visual Segmentation? A Study from Supervised and Zero-shot Perspectives](https://proceedings.bmvc2023.org/367/)", published at [BMVC 2023](https://proceedings.bmvc2023.org).
This work mainly explores two-part benefits of contrastive pre-training for audio-visual segmentation (AVS). 
1. Zero-shot setting. In this setting, the pre-trained models work together with the Segment Anything Model (SAM) to achieve Zero-shot AVS.
2. Supervised setting. In this setting, the work mainly explores how much the segmentor can gain when using contrastively pre-trained model weights to init the backbone.

## Dictionary Structure
* `./environment_config` the yml files of conda environments used in the work.

## Environment Preparation
### Zero-shot Setting
```bash
conda env create -n ZSAVS -f Zero_shot_AVS.yml
```
### Supervised Setting
```bash
conda env create -n audio_seg -f audio_seg_config.yml
```

## Dataset Preparation
For the dataset, we follow [AVSBench, ECCV 2022](https://opennlplab.github.io/AVSBench/). You can also access data via our [google drive](https://drive.google.com/drive/folders/1vLe-f2uoA-FV2qR2ZF4zawW9eYGCQ_xc?usp=drive_link)

## Pretrained Model Weights Preparation
TODO



## Zero-shot AVS
TODO

## Validation of Contrastive Pre-training Benefit for AVS
TODO

