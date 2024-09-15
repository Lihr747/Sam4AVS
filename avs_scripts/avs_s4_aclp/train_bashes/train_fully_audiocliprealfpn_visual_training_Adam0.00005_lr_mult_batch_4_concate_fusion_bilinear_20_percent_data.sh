export CUDA_VISIBLE_DEVICES=1

nohup python train_partial_data.py --session_name S4_$(basename $0) --rest_frac 0.2 --visual_backbone audioclip_realfpn --train_batch_size 4 --lr 0.00005 \
--freeze_audio_backbone --backbone_lr_mult 0.1 --max_epoches 70 --fusion concate --fpn_upmode bilinear > /dev/null 2>&1 &
# for fusion we use 1024 7 7 to replace 2048 7 7 > /dev/