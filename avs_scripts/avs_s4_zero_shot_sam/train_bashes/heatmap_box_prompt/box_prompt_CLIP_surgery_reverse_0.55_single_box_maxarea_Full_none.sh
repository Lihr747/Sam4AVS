filename=${0##*/}
filename=${filename%.*}
echo $filename

export CUDA_VISIBLE_DEVICES=0

base_folder=/home/yujr/workstation/Audio-Visual-Seg/avsbench_data/AVS_ZS_results/$filename
mkdir $base_folder

nohup python test_heatmap_prompt_box.py --log_dir $base_folder --point_strategy single_maxarea --backbone CLIP_Surgery --reverse --noise none --thres 0.55 --model_type Full > $base_folder/res.out &
