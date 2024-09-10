export CUDA_VISIBLE_DEVICES=4

nohup python train_weights_ablation.py --session_name S4_$(basename $0) --visual_backbone audioclip_realfpn_weights_ablation --train_batch_size 4 --lr 0.00005 \
--freeze_audio_backbone --backbone_lr_mult 0.1 --max_epoches 70 --fusion concate --fpn_upmode bilinear --audioclip_path "/home/yujr/workstation/Audio-Visual-Seg/pretrained_backbones/resnet50-19c8e357.pth,/home/yujr/workstation/Audio-Visual-Seg/pretrained_backbones/AudioCLIP-Full-Training.pt" > /dev/null 2>&1 &