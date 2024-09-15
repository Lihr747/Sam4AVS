import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from config import cfg
from dataloader_point_prompt_multi import MS3Dataset
from torchvision.ops import box_convert
import copy

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb
from model import AudioCLIP, AudioCLIP_Surgery

import cv2
from segment_anything import build_sam, SamPredictor
import torchvision as tv
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###########################################
# Load sam AudioCLIP
# optional: /home/yujr/workstation/Audio-Visual-Seg/pretrained_backbones/AudioCLIP-Partial-Training.pt

sam_checkpoint = '../../pretrained_backbones/sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

SAMPLE_RATE = 44100
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

image_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    tv.transforms.CenterCrop(IMAGE_SIZE),
    tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
])
audio_transforms = utils.transforms.ToTensor1D()


def show_heatmap(img, similarity):
    img = cv2.resize(img, (224, 224))
    heatmap = similarity
    vis = (heatmap * 255).astype('uint8')
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis = img * 0.6 + vis * 0.4
    vis = vis.astype('uint8')
    return vis

######
## gen a noise
def gen_noise(noise, percent=0.1):
    if noise == 'rand':
        np.random.seed(0)
        random_values = np.random.rand(220500)
        return torch.tensor(percent * random_values).float().unsqueeze(0).cuda()
    elif noise == 'zero':
        random_values = np.zeros(220500)
        return torch.tensor(percent * random_values).float().unsqueeze(0).cuda()
    elif noise == 'gausan':
        np.random.seed(0)
        random_values = np.random.normal(0, 1, 220500)
        return torch.tensor(percent * random_values).float().unsqueeze(0).cuda()
######

######
## get similarity score
def get_similarity_score(audio_feature, redundant_feature, visual_features, reverse=True):
    audio_feature = audio_feature - redundant_feature
    region_feature = visual_features[0][-1]
    region_feature = region_feature / region_feature.norm(dim=1, keepdim=True)
    region_feature = region_feature.squeeze(0).reshape(-1, 49)
    similarity = audio_feature @ region_feature
    if reverse:
        similarity = 1 - similarity
    sm = (similarity - similarity.min()) / (similarity.max() - similarity.min())
    sm = sm.reshape((1, 7, 7))
    sm = sm.unsqueeze(0)
    sm = torch.nn.functional.interpolate(sm, (IMAGE_SIZE, IMAGE_SIZE), mode='bilinear')
    sm = sm.squeeze(0).squeeze(0)
    sm = sm.cpu().numpy()
    return sm
######

######
## get topk points
def get_topk_points(similarity):
    # pos points
    similarity = similarity.reshape((IMAGE_SIZE* IMAGE_SIZE))
    index = similarity.argmax()
    points = []
    labels = []
    points.append([index % IMAGE_SIZE + 0.5, index // IMAGE_SIZE + 0.5])
    labels.append(1)
    # neg points
    index = similarity.argmin()
    points.append([index % IMAGE_SIZE + 0.5, index // IMAGE_SIZE + 0.5])
    labels.append(0)
    input_point = np.array(points)
    input_label = np.array(labels)
    return input_point, input_label
    
######

######
## get peak points
def get_peak_points(similarity, downsample=7, t=0.7):
    #similarity = cv2.GaussianBlur(similarity, (3, 3), 0)
    similarity_with_borders = np.pad(similarity, [(2, 2), (2, 2)], mode='constant')
    similarity_center = similarity_with_borders[1:similarity_with_borders.shape[0] - 1, 1:similarity_with_borders.shape[1] - 1]
    similarity_left = similarity_with_borders[1:similarity_with_borders.shape[0] - 1, 2:similarity_with_borders.shape[1]]
    similarity_right = similarity_with_borders[1:similarity_with_borders.shape[0] - 1, 0:similarity_with_borders.shape[1] - 2]
    similarity_up = similarity_with_borders[2:similarity_with_borders.shape[0], 1:similarity_with_borders.shape[1] - 1]
    similarity_down = similarity_with_borders[0:similarity_with_borders.shape[0] - 2, 1:similarity_with_borders.shape[1] - 1]
    pos_peaks = (similarity_center > similarity_left) & \
                    (similarity_center > similarity_right) & \
                    (similarity_center > similarity_up) & \
                    (similarity_center > similarity_down)
    neg_peaks = (similarity_center < similarity_left) & \
                    (similarity_center < similarity_right) & \
                    (similarity_center < similarity_up) & \
                    (similarity_center < similarity_down)
    try_pos_peaks = pos_peaks & (similarity_center > t)
    if sum(try_pos_peaks.reshape(-1)) == 0:
        try_pos_peaks = similarity_center == similarity_center.max()
    pos_peaks = try_pos_peaks
    try_neg_peaks = neg_peaks & (similarity_center < 1 - t)
    if sum(try_neg_peaks.reshape(-1)) == 0:
        try_neg_peaks = similarity_center == similarity_center.min()
    neg_peaks = try_neg_peaks
    pos_peaks = pos_peaks[1:similarity_center.shape[0] - 1, 1:similarity_center.shape[1] - 1]
    neg_peaks = neg_peaks[1:similarity_center.shape[0] - 1, 1:similarity_center.shape[1] - 1]
    pos_points = np.argwhere(pos_peaks)
    neg_points = np.argwhere(neg_peaks)
    points = np.concatenate([pos_points, neg_points], axis=0)
    input_points = points
    input_points[:, 0] = points[:, 1]
    input_points[:, 1] = points[:, 0]
    input_labels = np.concatenate([np.ones(pos_points.shape[0]), np.zeros(neg_points.shape[0])], axis=0)
    return points, input_labels

######

######
## get dense points
def get_dense_points(similarity, downsample=7, t=0.8):
    similarity = similarity.reshape((1, 1, IMAGE_SIZE, IMAGE_SIZE))
    down_side = IMAGE_SIZE // downsample
    similarity = torch.tensor(similarity)
    similarity = torch.nn.functional.interpolate(similarity, (down_side, down_side), mode='bilinear')[0, 0, :, :]
    similarity = similarity.cpu().numpy()
    h, w = similarity.shape
    scale_h = IMAGE_SIZE / h
    scale_w = IMAGE_SIZE / w
    similarity = similarity.reshape((h * w))
    rank = similarity.argsort()
    num = max(1, min((similarity >= t).sum(), similarity.shape[0] // 2))
    points = []
    labels = np.ones(num * 2).astype('uint8')
    labels[num:] = 0
    points = []
    # pos points
    for index in rank[-num:]:
        points.append([(index % h + 0.5) * scale_h, (index // w + 0.5) * scale_w])
    # neg points
    for index in rank[:num]:
        points.append([(index % h + 0.5) * scale_h, (index // w + 0.5) * scale_w])
    input_point = np.array(points)
    input_label = labels
    return input_point, input_label
######

def myfn(batch_list):
    image_paths = []
    for i in range(len(batch_list)):
        image_paths += batch_list[i][0]
    audio = torch.stack([item[1] for item in batch_list])
    mask = torch.stack([item[2] for item in batch_list])
    video_name = [item[3] for item in batch_list]

    return image_paths, audio, mask, video_name


###########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='/home/yujr/workstation/Audio-Visual-Seg/avsbench_data/AVS_ZS_results/debug_multi', type=str)
    parser.add_argument('--point_strategy', default='peak', type=str)
    parser.add_argument('--backbone', default='Surgery', type=str)
    parser.add_argument('--reverse', default=False, action='store_true', help='Bool type')
    parser.add_argument('--multi_output', default=True, type=bool)
    parser.add_argument('--selection_method', default='max_score', type=str)
    parser.add_argument('--noise', default='rand', type=str)
    parser.add_argument('--thres', default=0.8, type=float)
    parser.add_argument('--model_type', default='Partial', type=str)

    # Test data
    args = parser.parse_args()
    split = 'test'
    test_dataset = MS3Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        collate_fn=myfn,
                                                        pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    if args.noise != 'none':
        redundant_noise = gen_noise(args.noise)
        redundant_noise = redundant_noise.to(device=device)

    if args.backbone == 'CLIP':
        aclp = AudioCLIP(pretrained='../../pretrained_backbones/AudioCLIP-{}-Training.pt'.format(args.model_type))
    else:
        aclp = AudioCLIP_Surgery(pretrained='../../pretrained_backbones/AudioCLIP-{}-Training.pt'.format(args.model_type))
    aclp.to(device=device)
    aclp.eval()
    visual_encoder = aclp.visual
    audio_encoder = aclp.audio

    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            image_paths, audio, masks, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            # imgs is transformed !!! NEED add more preprocess to process for DINO

            masks = masks.to(device=device)
            B, frame, C, H, W = masks.shape
            masks = masks.view(B*frame, H, W)

            audio = audio.to(device=device)
            audio = audio.squeeze(0)
            audio_features = aclp.encode_audio(audio=audio)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
            if args.noise == 'none':
                redundant_feature = torch.zeros_like(audio_features[0]).to(device=device)
            else:
                redundant_feature = aclp.encode_audio(audio=redundant_noise)
                redundant_feature = redundant_feature / redundant_feature.norm(dim=-1, keepdim=True)

            pred_masks = []
            visual_heatmap = []

            for i in range(len(image_paths)):
                pil_image = Image.open(image_paths[i]).convert('RGB')
                pil_image = image_transforms(pil_image)
                pil_image = pil_image.unsqueeze(0)
                pil_image = pil_image.to(device=device)
                audio_feature = audio_features[i].reshape(1, -1)
                visual_features = aclp.encode_image(pil_image)
                similarity = get_similarity_score(audio_feature, redundant_feature, visual_features, args.reverse)

                
                ## get point
                if args.point_strategy == 'top':
                    points, labels = get_topk_points(similarity)
                elif args.point_strategy == 'peak':
                    points, labels = get_peak_points(similarity, t = args.thres)
                elif args.point_strategy == 'dense':
                    points, labels = get_dense_points(similarity, t = args.thres)
                else:
                    raise NotImplementedError
                image = cv2.imread(image_paths[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                vis = show_heatmap(image, similarity)
                sam_predictor.set_image(image)
                sam_pred_masks, scores, logits = sam_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=args.multi_output,
                )
                if args.multi_output == False:
                    pred_mask = sam_pred_masks[0]
                else:
                    if args.selection_method == 'max_area':
                        bool_masks = sam_pred_masks.reshape((3, -1))
                        area = bool_masks.sum(axis=1)
                        index = area.argmax()
                        pred_mask = sam_pred_masks[index]
                    elif args.selection_method == 'max_score':
                        index = scores.argmax()
                        pred_mask = sam_pred_masks[index]
                pred_masks.append(torch.tensor(pred_mask).to(device=device))
                visual_heatmap.append(vis)

            pred_masks = torch.stack(pred_masks)
            visual_heatmap = [visual_heatmap]

            mask_save_path = os.path.join(args.log_dir, 'pred_masks')
            save_mask(pred_masks, mask_save_path, video_name_list, vis_raw_img=True, raw_img_path=cfg.DATA.DIR_IMG, visual_heatmap=visual_heatmap)

            miou = mask_iou(pred_masks, masks)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(pred_masks, masks, args.log_dir)
            avg_meter_F.add({'F_score': F_score})
            print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))
            # /home/yujr/workstation/Audio-Visual-Seg/avsbench_data/train_logs/aclp_s4_logs/S4_train_fully_audiocliprealfpn_visual_training_Adam0.0001_lr_mult.sh_20230331-064343/checkpoints/S4_train_fully_audiocliprealfpn_visual_training_Adam0.0001_lr_mult.sh_best.pth

        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
        print('test miou: {}, F_score: {}'.format(miou.item(), F_score))