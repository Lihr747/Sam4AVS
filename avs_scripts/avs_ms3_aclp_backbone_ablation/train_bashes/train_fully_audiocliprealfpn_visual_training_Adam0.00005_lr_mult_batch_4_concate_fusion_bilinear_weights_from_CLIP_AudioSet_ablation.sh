export CUDA_VISIBLE_DEVICES=2

nohup python train_weights_ablation.py --session_name MS3_$(basename $0) --visual_backbone audioclip_realfpn_weights_ablation --train_batch_size 4 --lr 0.00005 \
--freeze_audio_backbone --backbone_lr_mult 0.1 --max_epoches 200 --fusion concate --fpn_upmode bilinear --audioclip_path "/home/yujr/workstation/Audio-Visual-Seg/pretrained_backbones/CLIP.pt,/home/yujr/workstation/Audio-Visual-Seg/pretrained_backbones/ESRNXFBSP.pt" > /dev/null 2>&1 &
# for fusion we use 1024 7 7 to replace 2048 7 7 > /dev/