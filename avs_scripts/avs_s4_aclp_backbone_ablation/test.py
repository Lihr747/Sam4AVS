import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument('--fusion', default='skip', type=str, help='optional: skip, concat')
    parser.add_argument('--fpn_upmode', default='nearest', type=str, help='optional: nearest, bilinear')

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')

    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='../../avsbench_data/train_logs/aclp_s4_logs', type=str)

    args = parser.parse_args()

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    elif (args.visual_backbone).lower() == "audioclip":
        from model import ACLP_AVSModel as AVSModel
        print('==> Use AudioCLIP as the visual backbone...')
    elif (args.visual_backbone).lower() == "audioclip_realfpn":
        from model import ACLP_AVSModel_real_Semantic_FPN as AVSModel
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")


    # Log directory
    if not os.path.exists(args.log_dir):
        raise FileNotFoundError

    # Set logger
    log_path = os.path.join(args.log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'test.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        args=args, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)

    ckpt_folder = os.path.join(args.log_dir, 'checkpoints')
    args.weights = '/home/yujr/workstation/Audio-Visual-Seg/avsbench_data/train_logs/aclp_s4_logs/S4_train_fully_audiocliprealfpn_visual_training_Adam0.00005_lr_mult_batch_4_concate_fusion_bilinear.sh_20230502-075450/checkpoints/S4_train_fully_audiocliprealfpn_visual_training_Adam0.00005_lr_mult_batch_4_concate_fusion_bilinear.sh_best.pth'
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    logger.info('=> Load trained model %s'%args.weights)

    # audio backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Test data
    split = 'test'
    test_dataset = S4Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    model.eval()
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, category_list, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B*frame, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3])

            output, _, _ = model(imgs, audio) # [5, 1, 224, 224] = [bs=1 * T=5, 1, 224, 224]
            if args.save_pred_mask:
                mask_save_path = os.path.join(args.log_dir, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path, category_list, video_name_list, vis_raw_img=True, raw_img_path=os.path.join(cfg.DATA.DIR_IMG, split))

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(output.squeeze(1), mask, args.log_dir)
            avg_meter_F.add({'F_score': F_score})
            logger.info(category_list[0] + ',' + video_name_list[0])
            logger.info('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))
            # /home/yujr/workstation/Audio-Visual-Seg/avsbench_data/train_logs/aclp_s4_logs/S4_train_fully_audiocliprealfpn_visual_training_Adam0.0001_lr_mult.sh_20230331-064343/checkpoints/S4_train_fully_audiocliprealfpn_visual_training_Adam0.0001_lr_mult.sh_best.pth

        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))












