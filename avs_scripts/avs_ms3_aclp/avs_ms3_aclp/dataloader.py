import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms
import librosa
from config import cfg
import pdb



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
    track, _ = librosa.load(audio_wav_path, sr=cfg.DATA.SAMPLE_RATE, dtype=np.float32)
    #track = track.detach() # [220500] TODO: => [5, XXXX]
    # 30 of 4925 less than 5s, TODO: padding last 1s to 5s
    MAX_LENGTH = 5 * cfg.DATA.SAMPLE_RATE
    if track.shape[0] > MAX_LENGTH:
        track = track[:MAX_LENGTH]
    elif track.shape[0] < MAX_LENGTH:
        second = track.shape[0] // cfg.DATA.SAMPLE_RATE
        rest_second = 5 - second
        last_second_feature = track[-cfg.DATA.SAMPLE_RATE:]
        track = track[0:second*cfg.DATA.SAMPLE_RATE]
        track = np.concatenate((track, np.tile(last_second_feature, rest_second)))
    track = track.reshape(1, -1)
    track = transform(track)
    track = track.reshape(5, 1, -1) # 5 x 1 x 44100
    track = track.repeat((1, 1, 5))
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
            transforms.Resize(cfg.DATA.IMAGE_SIZE, interpolation=Image.BICUBIC),
            transforms.CenterCrop(cfg.DATA.IMAGE_SIZE),
            transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.audio_transform = transforms.Compose([
            transforms.ToTensor(),
        ])



    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        audio_wav_path = os.path.join(cfg.DATA.DIR_AUDIO, self.split, video_name + '.wav')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        # audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        audio_tensor = load_audio_wav(audio_wav_path, self.audio_transform)
        
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_tensor, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)

class MS3Dataset_partial(Dataset):
    """Dataset for multiple sound source segmentation"""
    def __init__(self, split='train', rest_frac=0.5):
        super(MS3Dataset_partial, self).__init__()
        self.split = split
        self.mask_num = 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        self.df_split = self.df_split.sample(frac=rest_frac, replace=False, random_state=0)
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(cfg.DATA.IMAGE_SIZE, interpolation=Image.BICUBIC),
            transforms.CenterCrop(cfg.DATA.IMAGE_SIZE),
            transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.audio_transform = transforms.Compose([
            transforms.ToTensor(),
        ])



    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        audio_wav_path = os.path.join(cfg.DATA.DIR_AUDIO, self.split, video_name + '.wav')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        # audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        audio_tensor = load_audio_wav(audio_wav_path, self.audio_transform)
        
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_tensor, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)






if __name__ == "__main__":
    train_dataset = MS3Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()