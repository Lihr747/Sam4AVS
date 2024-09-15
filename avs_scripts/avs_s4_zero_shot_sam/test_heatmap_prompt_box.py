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
from dataloader_point_prompt import S4Dataset
from torchvision.ops import box_convert
import copy

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask, save_mask_box
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

def show_box(img, boxes):
    img = cv2.resize(img, (224, 224))
    vis = img.copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    for box in boxes:
        vis = cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
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

## get boxes
def get_multi_box(sm, thres=0.8):
    similarity = copy.deepcopy(sm)
    similarity[similarity >= thres] = 1.0
    similarity[similarity < thres] = 0.0
    similarity = np.int8(similarity)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(similarity, connectivity=8)
    # locate all 0 mask index
    zero_index = np.where(similarity == 0)
    remove_index = set(labels[zero_index])
    # remove 0 mask index
    new_index = [i for i in range(retval) if i not in remove_index]
    boxes = np.hstack((stats[new_index, 0:2], stats[new_index, 0:2] + stats[new_index, 2:4] - 1))
    return boxes

## get boxes
def get_single_box_maxarea(sm, thres=0.8):
    similarity = copy.deepcopy(sm)
    similarity[similarity >= thres] = 1.0
    similarity[similarity < thres] = 0.0
    similarity = np.int8(similarity)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(similarity, connectivity=8)
    # locate all 0 mask index
    zero_index = np.where(similarity == 0)
    remove_index = set(labels[zero_index])
    new_index = [i for i in range(retval) if i not in remove_index]
    new_stats = stats[new_index]
    area = new_stats[:, 4]
    max_area_index = area.argmax()
    return np.hstack((new_stats[max_area_index, 0:2], new_stats[max_area_index, 0:2] + new_stats[max_area_index, 2:4] - 1))[np.newaxis,]

## get boxes
def get_single_box_maxscore(sm, thres=0.8):
    similarity = copy.deepcopy(sm)
    similarity[similarity >= thres] = 1.0
    similarity[similarity < thres] = 0.0
    similarity = np.int8(similarity)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(similarity, connectivity=8)
    sm = sm.reshape((IMAGE_SIZE* IMAGE_SIZE))
    index = sm.argmax()
    stat_index = labels[index // IMAGE_SIZE, index % IMAGE_SIZE]
    return np.hstack((stats[stat_index, 0:2], stats[stat_index, 0:2] + stats[stat_index, 2:4] - 1))[np.newaxis,]


## get boxes
def get_total_box(sm, thres=0.8):
    similarity = copy.deepcopy(sm)
    similarity[similarity >= thres] = 1.0
    similarity[similarity < thres] = 0.0
    similarity = np.int8(similarity)
    y_coords, x_coords = np.nonzero(similarity)
    x_min = x_coords.min()  
    x_max = x_coords.max()  
    y_min = y_coords.min()  
    y_max = y_coords.max()
    return np.array([x_min, y_min, x_max, y_max])[np.newaxis,]

def myfn(batch_list):
    image_paths = []
    for i in range(len(batch_list)):
        image_paths += batch_list[i][0]
    audio = torch.stack([item[1] for item in batch_list])
    mask = torch.stack([item[2] for item in batch_list])
    category = [item[3] for item in batch_list]
    video_name = [item[4] for item in batch_list]

    return image_paths, audio, mask, category, video_name


###########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='/home/yujr/workstation/Audio-Visual-Seg/avsbench_data/AVS_ZS_results/debug', type=str)
    parser.add_argument('--point_strategy', default='multi_box', type=str)
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
    test_dataset = S4Dataset(split)
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
            image_paths, audio, masks, category_list, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
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
            box_visualization = []

            for i in range(len(image_paths)):
                pil_image = Image.open(image_paths[i]).convert('RGB')
                pil_image = image_transforms(pil_image)
                pil_image = pil_image.unsqueeze(0)
                pil_image = pil_image.to(device=device)
                audio_feature = audio_features[i].reshape(1, -1)
                visual_features = aclp.encode_image(pil_image)
                similarity = get_similarity_score(audio_feature, redundant_feature, visual_features, args.reverse)

                
                ## get point
                if args.point_strategy == 'multi_box':
                    boxes = get_multi_box(similarity, args.thres)
                elif args.point_strategy == 'total_box':
                    boxes = get_total_box(similarity, args.thres)
                elif args.point_strategy == 'single_maxscore':
                    boxes = get_single_box_maxscore(similarity, args.thres)
                elif args.point_strategy == 'single_maxarea':
                    boxes = get_single_box_maxarea(similarity, args.thres)
                else:
                    raise NotImplementedError
                image = cv2.imread(image_paths[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                vis = show_heatmap(image, similarity)
                box_vis = show_box(image, boxes)
                sam_predictor.set_image(image)
                total_pred_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE)).astype(np.bool8)
                
                for box in boxes:
                    sam_pred_masks, scores, logits = sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box = box[None, :],
                        multimask_output=args.multi_output,
                    )
                    index = scores.argmax()
                    pred_mask = sam_pred_masks[index]
                    total_pred_mask = np.bitwise_or(total_pred_mask, pred_mask)
                pred_mask = total_pred_mask

                pred_masks.append(torch.tensor(pred_mask).to(device=device))
                visual_heatmap.append(vis)
                box_visualization.append(box_vis)

            pred_masks = torch.stack(pred_masks)
            visual_heatmap = [visual_heatmap]
            box_visualization = [box_visualization]

            mask_save_path = os.path.join(args.log_dir, 'pred_masks')
            save_mask_box(pred_masks, mask_save_path, category_list, video_name_list, vis_raw_img=True, raw_img_path=os.path.join(cfg.DATA.DIR_IMG, split), visual_heatmap=visual_heatmap, box_vis=box_visualization)

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