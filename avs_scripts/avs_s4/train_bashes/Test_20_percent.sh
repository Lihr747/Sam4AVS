export CUDA_VISIBLE_DEVICES=1

nohup python test.py --session_name TPAVI_S4_$(basename $0) --visual_backbone resnet --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --log_dir /home/yujr/workstation/Audio-Visual-Seg/avsbench_data/train_logs/s4_logs \
--weights "/home/yujr/workstation/Audio-Visual-Seg/avsbench_data/train_logs/s4_logs/TPAVI_S4_TPAVI_partial_test_20_percent.sh_20230716-115511/checkpoints/TPAVI_S4_TPAVI_partial_test_20_percent.sh_best.pth" > test.out & 
#/home/yujr/workstation/Audio-Visual-Seg/avsbench_data/train_logs/s4_logs/TPAVI_S4_TPAVI_partial_test_50_percent.sh_20230716-115515/checkpoints/TPAVI_S4_TPAVI_partial_test_50_percent.sh_best.pth
