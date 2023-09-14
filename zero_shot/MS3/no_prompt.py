import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
from PIL import Image, ImageDraw
from torchvision import transforms
from config import cfg
from dataloader import MS3Dataset

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb
from model import AudioCLIP

import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###########################################

def myfn(batch_list):
    image_paths = []
    for i in range(len(batch_list)):
        image_paths += batch_list[i][0]
    audio = torch.stack([item[1] for item in batch_list])
    mask = torch.stack([item[2] for item in batch_list])
    video_name = [item[3] for item in batch_list]

    return image_paths, audio, mask, video_name

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def preprocess_img(image):
    transform = transforms.Compose([ 
                    transforms.ToTensor(), 
                    transforms.Resize(cfg.DATA.IMAGE_SIZE, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(cfg.DATA.IMAGE_SIZE),
                    transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
                    ])
    
    image = image.convert(mode='RGB')
    img_tensor = transform(image)
    return img_tensor

def retriev(elements, audio_features):
    preprocessed_images = []
    for image in elements:
        h, w = image.size
        if h == 0 or w == 0:
            continue
        preprocessed_images.append(preprocess_img(image).to(device))
    stacked_images = torch.stack(preprocessed_images)
    _, image_features = visual_encoder(stacked_images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    audio_features /= audio_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ audio_features.T
    scores = probs[:, 0].softmax(dim=0)
 
    return scores

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

###########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument('--save_path', default='../../avsbench_data/save_masks/ms3_audioclip_image_retrival', type=str)

    # Test data
    args = parser.parse_args()
    split = 'test'

    aclp = AudioCLIP(pretrained='../../pretrained_backbones/AudioCLIP-Full-Training.pt')
    sam = sam_model_registry["default"](checkpoint="../../pretrained_backbones/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    aclp.eval()
    aclp.to(device=device)
    visual_encoder = aclp.visual
    audio_encoder = aclp.audio

    test_dataset = MS3Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        collate_fn=myfn,
                                                        pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            image_paths, audio, masks, video_name_list = batch_data 
          
            audio = audio.to(device=device)
            masks = masks.to(device=device)
            B, frame, C, H, W = masks.shape
            masks = masks.view(B*frame, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3])
            audio_features = audio_encoder(audio)

            pred_masks = []
            for i in range(len(image_paths)):
                image_path = image_paths[i]
                audio_feature = audio_features[i].reshape(1, -1) 
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                generate_masks = mask_generator.generate(image)
                image = Image.open(image_path)
                cropped_boxes = []

                for mask in generate_masks:
                    cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))

                scores = retriev(cropped_boxes, audio_feature)
                indices = get_indices_of_values_above_threshold(scores, 0.5)

                pred_mask = torch.zeros([H, W])

                mask_list = [torch.from_numpy(generate_masks[indice]["segmentation"]).to(torch.float32) for indice in indices]
                for mask_obj in mask_list:
                    pred_mask = 1 - (1 - pred_mask) * (1 - mask_obj)
                pred_masks.append(pred_mask)
            pred_masks = torch.stack(pred_masks).to(device)
                        
            save_mask(pred_masks, masks, args.save_path, video_name_list, vis_raw_img=True, raw_img_path=cfg.DATA.DIR_IMG)

            miou = mask_iou(pred_masks, masks)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(pred_masks, masks, args.save_path)
            avg_meter_F.add({'F_score': F_score})
            print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))
           
        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))












