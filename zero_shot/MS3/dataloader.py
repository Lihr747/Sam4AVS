import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle
import librosa
import json
import cv2
from PIL import Image
from torchvision import transforms

from config import cfg
import pdb

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()# [5, 1, 96, 64]
    return audio_log_mel

def load_audio_wav(audio_wav_path, transform=None):
    sr = 48000
    track, _ = librosa.load(audio_wav_path, sr=sr)
    #track = track.detach() # [220500] TODO: => [5, XXXX]
    # 30 of 4925 less than 5s, TODO: padding last 1s to 5s
    MAX_LENGTH = 5 * sr
    if track.shape[0] > MAX_LENGTH:
        track = track[:MAX_LENGTH]
    elif track.shape[0] < MAX_LENGTH:
        second = track.shape[0] // sr
        rest_second = 5 - second
        last_second_feature = track[-sr:]
        track = track[0:second*sr]
        track = np.concatenate((track, np.tile(last_second_feature, rest_second)))
    track = torch.from_numpy(int16_to_float32(float32_to_int16(track))).float()
    track = track.reshape(1, -1)
    track = track.reshape(5, 1, -1) # 5 x 1 x 44100
    return track


class MS3Dataset(Dataset):
    """Dataset for multiple sound source segmentation"""
    def __init__(self, split='train'):
        super(MS3Dataset, self).__init__()
        self.split = split
        self.mask_num = 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
  
    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        audio_wav_path = os.path.join(cfg.DATA.DIR_AUDIO, self.split, video_name + '.wav')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        img_paths, masks = [], []
        for img_id in range(1, 6):
            img_path = os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id))
            img_paths.append(img_path)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        audio_tensor = load_audio_wav(audio_wav_path)
        masks_tensor = torch.stack(masks, dim=0)

        return img_paths, audio_tensor, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)


if __name__ == "__main__":
    test_dataset = MS3Dataset('test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=8,
                                                        pin_memory=True)

    for n_iter, batch_data in enumerate(test_dataloader):
        image_paths, audio, masks, video_name_list = batch_data 

        B, frame, C, H, W = masks.shape
        size = H * W
        masks = masks.view(B*frame, H, W)
        for i in range(B*frame):
            mask = masks[i]
            mask_rate = mask.sum().item() / size

            print(mask_rate)
        

        

