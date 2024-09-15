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
### Dataset Download
For the dataset, we follow [AVSBench, ECCV 2022](https://opennlplab.github.io/AVSBench/). You can also access data via our [google drive](https://drive.google.com/drive/folders/1vLe-f2uoA-FV2qR2ZF4zawW9eYGCQ_xc?usp=drive_link). Then, put the data into the `avsbench_data` folder.

### Dataset Preprocessing
```bash
python preprocess_s4.py
python preprocess_ms3.py
```

## Pretrained Model Weights Preparation
Pretrained models we used in this work include `AudioCLIP`, `CLIP`, `ESResNeXt`, `SAM`, `VGGish`, `resnet50` and `PVT`.

For `AudioCLIP`, `CLIP` and `ESResNeXt`, you can access the pre-trained checkpoints from [AudioCLIP releases](https://github.com/AndreyGuzhov/AudioCLIP/releases).

For `SAM`, you can access the pre-trained checkpoint from [SAM-VIT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

For `VGGish`, `resnet50` and `PVT`, we follow [AVSbench pretrained backbones](https://github.com/OpenNLPLab/AVSBench/tree/main/pretrained_backbones)

## Directory Tree after Dataset and Model Weights Preparation

```text
Sam4AVS
├─ avs_scripts
 ├─ avs_ms3
 ├─ avs_ms3_aclp
 ├─ avs_s4
 ├─ avs_s4_aclp
 ├─ avs_ms3_aclp_ablation
 ├─ avs_s4_aclp_ablation
 └─ avs_s4_zero_shot_sam
├─ avsbench_data
 ├─ Multi-sources
  └─ ms3_data
   ├─ audio_log_mel
   ├─ audio_wav
   ├─ gt_masks
   ├─ raw_videos
   └─ visual_frames
 ├─ Single-source
  └─ s4_data
   ├─ audio_log_mel
   ├─ audio_wav
   ├─ gt_masks
   ├─ raw_videos
   └─ visual_frames
 └─ train_logs
├─ preprocess_scripts
 ├─ preprocess_ms3.py
 └─ preprocess_s4.py
└─ pretrained_backbones
 ├─ AudioCLIP-Full-Training.pt
 ├─ AudioCLIP-Partial-Training.pt
 ├─ bpe_simple_vocab_16e6.txt.gz
 ├─ CLIP.pt
 ├─ ESRNXFBSP.pt
 ├─ pvt_v2_b5.pth
 ├─ resnet50-19c8e357.pth
 ├─ sam_vit_h_4b8939.pth
 └─ vggish-10086976.pth
```

## Supervised AVS
For single-source AVS

```bash
cd avs_scripts/avs_s4_aclp
bash train_bashes/train_fully_audiocliprealfpn_visual_training_Adam0.00005_lr_mult_batch_4_concate_fusion_bilinear.sh
```

For multi-source AVS

```bash
cd avs_scripts/avs_ms3_aclp
bash train_bashes/train_fully_audiocliprealfpn_visual_training_Adam0.00005_lr_mult_batch_4_concate_fusion_bilinear.sh
```

#### Validation of Contrastive Pre-training Benefit for AVS

For single-source AVS ablation

```bash
cd avs_scripts/avs_s4_aclp_ablation
bash train_bashes/*.sh
```

For multi-source AVS ablation

```bash
cd avs_scripts/avs_ms3_aclp_ablation
bash train_bashes/*.sh
```

## Zero-shot AVS

```bash
conda activate ZSAVS
```

### No-Prompt

For single-source AVS

```bash
cd ./zero_shot/S4
python no_prompt.py
```

For multi-source AVS

```bash
cd ./zero_shot/MS3
python no_prompt.py
```

### Point-Prompt

For single-source AVS

1. Point-prompt(local)

```bash
cd avs_scripts/avs_s4_zero_shot_sam
bash train_bashes/CLIP_surgery_reverse_0.6_peak_maxscore_Full_none.sh
```

2. Point-prompt(global)

```bash
cd avs_scripts/avs_s4_zero_shot_sam
bash train_bashes/CLIP_surgery_reverse_top_maxscore_Full_none.sh
```

3. Point-prompt(dense)

```bash
cd avs_scripts/avs_s4_zero_shot_sam
bash train_bashes/CLIP_surgery_reverse_0.85_dense_maxscore_Full_none.sh
```

For multi-source AVS

1. Point-prompt(local)

```bash
cd avs_scripts/avs_ms3_zero_shot_sam
bash train_bashes/Multi_CLIP_surgery_reverse_0.65_peak_maxscore_Full_none.sh
```

2. Point-prompt(global)

```bash
cd avs_scripts/avs_ms3_zero_shot_sam
bash train_bashes/Multi_CLIP_surgery_reverse_top_maxscore_Full_none.sh
```

3. Point-prompt(dense)

```bash
cd avs_scripts/avs_ms3_zero_shot_sam
bash train_bashes/Multi_CLIP_surgery_reverse_0.7_dense_maxscore_Full_none.sh
```


### Heatmap-based Box-Prompt

For single-source AVS

```bash
cd avs_scripts/avs_s4_zero_shot_sam
bash heatmap_box_prompt/box_prompt_CLIP_surgery_reverse_0.55_single_box_maxarea_Full_none.sh
```

For multi-source AVS

```bash
cd avs_scripts/avs_ms3_zero_shot_sam
bash heatmap_box_prompt/multi_box_prompt_CLIP_surgery_reverse_0.55_single_box_maxarea_Full_none.sh
```

### Box-Prompt

For single-source AVS

```bash
cd ./zero_shot/S4
python box_prompt.py
```

For multi-source AVS

```bash
cd ./zero_shot/MS3
python box_prompt.py
```
