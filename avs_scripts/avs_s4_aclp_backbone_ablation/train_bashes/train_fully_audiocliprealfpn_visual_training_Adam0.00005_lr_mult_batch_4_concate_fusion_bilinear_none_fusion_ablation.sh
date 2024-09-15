export CUDA_VISIBLE_DEVICES=1

nohup python train_weights_ablation.py --session_name S4_$(basename $0) --visual_backbone audioclip_realfpn_weights_ablation --train_batch_size 4 --lr 0.00005 \
--freeze_audio_backbone --backbone_lr_mult 0.1 --max_epoches 70 --fusion none --fpn_upmode bilinear > test.out &
# for fusion we use 1024 7 7 to replace 2048 7 7 > /dev/