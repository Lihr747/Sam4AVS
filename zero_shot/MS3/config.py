from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 4 # default 16
cfg.LAMBDA_1 = 5 # default: 5
cfg.MASK_NUM = 1 # 5 for fully supervised, 1 for weakly supervised

###############################
# TRAIN
cfg.TRAIN = edict()

cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "../../pretrained_backbones/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "../../pretrained_backbones/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "../../pretrained_backbones/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "../../pretrained_backbones/pvt_v2_b5.pth"

cfg.TRAIN.FINE_TUNE_SSSS = False
cfg.TRAIN.PRETRAINED_S4_aAVS_WO_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"
cfg.TRAIN.PRETRAINED_S4_AVS_WITH_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.DIR_JSON = "../../category_list"
cfg.DATA.ANNO_CSV = "../../avsbench_data//Multi-sources/ms3_meta_data.csv"
cfg.DATA.DIR_IMG = "../../avsbench_data//Multi-sources/ms3_data/visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = "../../avsbench_data//Multi-sources/ms3_data/audio_log_mel"
cfg.DATA.DIR_AUDIO = "../../avsbench_data//Multi-sources/ms3_data/audio_wav"
cfg.DATA.DIR_MASK = "../../avsbench_data//Multi-sources/ms3_data/gt_masks"
cfg.DATA.IMG_SIZE = (224, 224)
###############################


# def _edict2dict(dest_dict, src_edict):
#     if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
#         for k, v in src_edict.items():
#             if not isinstance(v, edict):
#                 dest_dict[k] = v
#             else:
#                 dest_dict[k] = {}
#                 _edict2dict(dest_dict[k], v)
#     else:
#         return


# def gen_config(config_file):
#     cfg_dict = {}
#     _edict2dict(cfg_dict, cfg)
#     with open(config_file, 'w') as f:
#         yaml.dump(cfg_dict, f, default_flow_style=False)


# def _update_config(base_cfg, exp_cfg):
#     if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
#         for k, v in exp_cfg.items():
#             if k in base_cfg:
#                 if not isinstance(v, dict):
#                     base_cfg[k] = v
#                 else:
#                     _update_config(base_cfg[k], v)
#             else:
#                 raise ValueError("{} not exist in config.py".format(k))
#     else:
#         return


# def update_config_from_file(filename):
#     exp_config = None
#     with open(filename) as f:
#         exp_config = edict(yaml.safe_load(f))
#         _update_config(cfg, exp_config)

if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()