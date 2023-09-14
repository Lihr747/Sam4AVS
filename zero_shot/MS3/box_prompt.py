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
from dataloader import MS3Dataset
from torchvision.ops import box_convert
import copy
import json

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb

import cv2
from segment_anything import build_sam, SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
import laion_clap
import supervision as sv
from huggingface_hub import hf_hub_download

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

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
    parser.add_argument('--save_path', default='../../avsbench_data/save_masks/ms3_clap_text_retrival', type=str)
    parser.add_argument("--text_ret_num", default=2, type=int)
    parser.add_argument("--dino_text_threshold", default=0.25, type=float)
    parser.add_argument("--dino_box_threshold", default=0.3, type=float)
    parser.add_argument('--category_list', default='../../category_list/audioset.json', type=str, help="category list from audioset")


    # Test data
    args = parser.parse_args()

    clap = laion_clap.CLAP_Module(enable_fusion=False)
    clap.load_ckpt()

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device=device)

    sam_checkpoint = '../../pretrained_backbones/sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

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

    with open(args.category_list,'rb') as f:
        categories = json.load(f)
    print("category number:", len(categories))

    text = [label for label in categories]

    text_features = clap.get_text_embedding(text, use_tensor=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            image_paths, audio, masks, video_name_list = batch_data 
            audio = audio.reshape(-1 , audio.shape[3])

            audio = audio.to(device=device)
            masks = masks.to(device=device)
            B, frame, C, H, W = masks.shape
            masks = masks.view(B*frame, H, W)
            audio_embed = clap.get_audio_embedding_from_data(x = audio, use_tensor=True)
            audio_embed /= audio_embed.norm(dim=-1, keepdim=True)

            pred_masks = []

            for i in range(len(image_paths)):
                image_path = image_paths[i]
                audio_features = audio_embed[i].reshape(1, -1)
                probs = text_features @ audio_features.T
                scores = probs[:, 0]
                conf, indices = scores.topk(args.text_ret_num)

                pred_mask = torch.zeros((H, W)).to(device)
                image_source, image = load_image(image_path)
                all_boxes = []

                for j in range(indices.shape[0]):
                    text_prompt = categories[indices[j]]

                    boxes, logits, phrases = predict(
                            model=groundingdino_model, 
                            image=image, 
                            caption=text_prompt, 
                            box_threshold=args.dino_box_threshold, 
                            text_threshold=args.dino_text_threshold
                    )

                    if boxes.shape[0] != 0 and boxes.shape[1] != 0:
                        all_boxes.append(boxes)

                if len(all_boxes) != 0:
                    all_boxes = torch.cat(all_boxes, dim=0)
                    sam_predictor.set_image(image_source)

                    H, W, _ = image_source.shape
                    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(all_boxes) * torch.Tensor([W, H, W, H])

                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
                    sam_masks, _, _ = sam_predictor.predict_torch(
                                        point_coords = None,
                                        point_labels = None,
                                        boxes = transformed_boxes,
                                        multimask_output = False,
                            )

                    for k in range(sam_masks.shape[0]):
                        sam_mask = sam_masks[k].reshape(H, W).to(torch.float32).to(device)
                        pred_mask = 1 - (1 - pred_mask) * (1 - sam_mask)
                    
                pred_masks.append(pred_mask)

            pred_masks = torch.stack(pred_masks)
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
        print('test miou: {}, F_score: {}'.format(miou.item(), F_score))












