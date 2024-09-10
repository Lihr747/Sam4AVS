import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import MS3Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb
from torch.utils.tensorboard import SummaryWriter

from model import AudioCLIP

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="MS3", type=str, help="the MS3 setting")
    parser.add_argument("--visual_backbone", default="audioclip", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=40, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--backbone_lr_mult", default=1.0, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument('--masked_av_flag', action='store_true', default=False, help='additional sa/masked_va loss for five frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing l4 loss')
    parser.add_argument("--masked_av_stages", default=[], nargs='+', type=int, help='compute sa/masked_va loss in which stages: [0, 1, 2, 3]')
    parser.add_argument('--threshold_flag', action='store_true', default=False, help='whether thresholding the generated masks')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')
    parser.add_argument('--norm_fea_flag', action='store_true', default=False, help='normalize audio-visual features')
    parser.add_argument('--closer_flag', action='store_true', default=False, help='use closer loss for masked_va loss')
    parser.add_argument('--euclidean_flag', action='store_true', default=False, help='use euclidean distance for masked_va loss')
    parser.add_argument('--kl_flag', action='store_true', default=False, help='use kl loss for masked_va loss')

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')
    parser.add_argument("--freeze_audio_backbone", action='store_true', default=False, help='whether to freeze the audio backbone')
    parser.add_argument("--freeze_visual_backbone", action='store_true', default=False, help='whether to freeze the visual backbone')
    parser.add_argument("--optimizer", default='Adam', type=str, help='optimizer: Adam or AdamW')

    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='../../avsbench_data/train_logs/aclp_ms3_logs', type=str)
    parser.add_argument('--fusion', default='skip', type=str, help='optional: skip, concat')
    parser.add_argument('--fpn_upmode', default='nearest', type=str, help='optional: nearest, bilinear')

    parser.add_argument("--not_save_pred_mask", action='store_true', default=False, help="save predited masks or not")

    parser.add_argument('--audioclip_path', default='../../pretrained_backbones/AudioCLIP-Full-Training.pt')

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


    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir
    writer = SummaryWriter(args.log_dir)

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        args=args, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag, \
                                        audioclip_path=args.audioclip_path)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    logger.info("==> Total params: %.2fM" % ( sum(p.numel() for p in model.parameters()) / 1e6))

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # audio_backbone = audio_extractor(cfg, device)
    # audio_backbone.cuda()
    # audio_backbone.eval()

    # Data
    train_dataset = MS3Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = MS3Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    # Optimizer
    conflicts = []
    if(args.freeze_visual_backbone):
        conflicts.append('module.aclp.visual')
    if(args.freeze_audio_backbone):
        conflicts.append('module.aclp.audio')
    head_model_params_norm = []
    backbone_model_params_norm = []
    head_model_params_no_norm = []
    backbone_model_params_no_norm = []
    for n, p in model.named_parameters():
        if not any([n.startswith(c) for c in conflicts]):
            if n.startswith('module.aclp'):
                if 'norm' in n:
                    backbone_model_params_norm.append(p)
                else:
                    backbone_model_params_no_norm.append(p)
            else:
                if 'norm' in n:
                    head_model_params_norm.append(p)
                else:
                    head_model_params_no_norm.append(p)
    # model_params = model.parameters()
    if(args.optimizer == 'Adam'):
        optimizer = torch.optim.Adam([{'params': head_model_params_norm + head_model_params_no_norm, 'lr': args.lr},
                            {'params': backbone_model_params_norm + backbone_model_params_no_norm, 'lr': args.lr * args.backbone_lr_mult}])
    elif(args.optimizer == 'AdamW'):
        optimizer = torch.optim.AdamW([{'params': head_model_params_no_norm, 'lr': args.lr, 'weight_decay': args.wt_dec},
                            {'params': backbone_model_params_no_norm, 'lr': args.lr * args.backbone_lr_mult, 'weight_decay': args.wt_dec},
                            {'params': head_model_params_norm, 'lr': args.lr, 'weight_decay': 0.0},
                            {'params': backbone_model_params_norm, 'lr': args.lr * args.backbone_lr_mult, 'weight_decay': 0.0}])
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')

    avg_meter_miou = pyutils.AverageMeter('miou')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(args.max_epoches):
        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, mask, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5 or 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask_num = 5
            mask = mask.view(B*mask_num, 1, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3]) # [B*T, 1, 96, 64]
            # with torch.no_grad():
            #     audio_feature = audio_backbone(audio) # [B*T, 128]

            output, visual_map_list, a_fea_list = model(imgs, audio) # [bs*5, 1, 224, 224]
            # TODO: modify IouSemanticAwareLoss params
            loss, loss_dict = IouSemanticAwareLoss(output, mask, a_fea_list, visual_map_list, \
                                        sa_loss_flag=args.masked_av_flag, lambda_1=args.lambda_1, count_stages=args.masked_av_stages, \
                                        mask_pooling_type=args.mask_pooling_type, threshold=args.threshold_flag, norm_fea=args.norm_fea_flag, \
                                        closer_flag=args.closer_flag, euclidean_flag=args.euclidean_flag, kl_flag=args.kl_flag)

            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})
            writer.add_scalar('total_loss', loss.item(), global_step=global_step)
            writer.add_scalar('iou_loss', loss_dict['iou_loss'], global_step=global_step)
            writer.add_scalar('sa_loss', loss_dict['sa_loss'], global_step=global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if (global_step-1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lambda_1:%.4f, lr: %.5f'%(
                        global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), args.lambda_1, optimizer.param_groups[0]['lr'])
                # train_log = ['Iter:%5d/%5d' % (global_step - 1, max_step),
                #         'Total_Loss:%.4f' % (avg_meter_total_loss.pop('total_loss')),
                #         'iou_loss:%.4f' % (avg_meter_L1.pop('iou_loss')),
                #         'sa_loss:%.4f' % (avg_meter_L4.pop('sa_loss')),
                #         'lambda_1:%.4f' % (args.lambda_1),
                #         'lr: %.4f' % (optimizer.param_groups[0]['lr'])]
                # print(train_log, flush=True)
                logger.info(train_log)

        # TODO: modify eval
        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                imgs, audio, mask, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B*frame, C, H, W)
                mask = mask.view(B*frame, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3])
                # with torch.no_grad():
                #     audio_feature = audio_backbone(audio)

                output, _, _ = model(imgs, audio) # [bs*5, 1, 224, 224]

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s'%model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            writer.add_scalar('Val_Miou', miou, global_step=global_step)
            # print(val_log)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
    # model test
    model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(model_save_path)
    for k, v in state_dict.items():
        name = 'module.'+k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    test_dataset = MS3Dataset('test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    model.eval()
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B*frame, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3]) # [B*T, 1, 96, 64]
            #with torch.no_grad():
            #audio_feature = None

            output, _, _ = model(imgs, audio) # [bs*5, 1, 224, 224]
            if not args.not_save_pred_mask:
                mask_save_path = os.path.join(log_dir, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path, video_name_list, vis_raw_img=True, raw_img_path=cfg.DATA.DIR_IMG)

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(output.squeeze(1), mask, args.log_dir)
            avg_meter_F.add({'F_score': F_score})
            logger.info('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))

        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))
        logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))









