import torch
import torch.nn as nn
import torchvision.models as models
from .audioclip import AudioCLIP
from model.TPAVI import TPAVIModule
import pdb
import numpy as np

class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x

class FPN_Neck(nn.Module):
    def __init__(self, in_channels, out_channels, start_level, end_level, up_sample_mode='nearest'):
        super(FPN_Neck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        self.up_sample_mode = up_sample_mode
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            l_conv = nn.Conv2d(self.in_channels[i], self.out_channels, kernel_size=1, stride=1, padding=0)
            fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += nn.functional.interpolate(
                laterals[i], scale_factor=2, mode=self.up_sample_mode)
        # build outputs
        outs = [ 
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        return tuple(outs)

class FPN_Head(nn.Module):
    def __init__(self, feature_strides, in_channels, out_channel, graph_size=32, class_size=1):
        super(FPN_Head, self).__init__()
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.graph_size = graph_size
        self.scale_heads = nn.ModuleList()
        for i in range(len(self.feature_strides)):
            head_length = max(1, int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Conv2d(self.in_channels[i] if k == 0 else self.out_channel, self.out_channel, kernel_size=1, stride=1, padding=0)
                )
                scale_head.append(
                    nn.GroupNorm(self.graph_size, self.out_channel)
                )
                scale_head.append(
                    nn.ReLU(inplace=True)
                )
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
                    )
            self.scale_heads.append(nn.Sequential(*scale_head))
        self.seg_conv = nn.Conv2d(self.out_channel, class_size, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        output = self.scale_heads[0](inputs[0])
        for i in range(1, len(self.feature_strides)):
            output = output + self.scale_heads[i](inputs[i])
        output = self.seg_conv(output)
        output = nn.functional.interpolate(output, scale_factor=4, mode='bilinear', align_corners=True)
        return output


class Pred_endecoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=256, config=None, tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True,
                 audioclip_path=None):
        super(Pred_endecoder, self).__init__()
        self.cfg = config
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag

        self.audioclip_path = audioclip_path

        self.aclp = AudioCLIP(pretrained=self.audioclip_path)
        self.visual_encoder = self.aclp.visual
        self.audio_encoder = self.aclp.audio
        self.relu = nn.ReLU(inplace=True)

        self.neck = FPN_Neck([256, 512, 1024, 3036], 256, 0, 4, up_sample_mode='nearest')
        self.head = FPN_Head([4, 8, 16, 32], [256, 256, 256, 256], 128, graph_size=32, class_size=1)


    def forward(self, x, audio=None):
        [x1, x2, x3, x4, last_feat], x_final = self.visual_encoder(x)
        audio_feature = self.audio_encoder(audio) # BF x 1024
        # x1: BF x 256  x 56 x 56
        # x2: BF x 512  x 28 x 28
        # x3: BF x 1024 x 14 x 14
        # x4: BF x 2048 x  7 x  7
        # last_feat: BF x 1024 x 7 x 7
        # x_final: BF x 1024

        # fuse audio feature to visual feature
        last_feat_H = last_feat.shape[2]
        last_feat_W = last_feat.shape[3]
        audio_feature = audio_feature.unsqueeze(2).unsqueeze(3) # BF x 1024 x 1 x 1
        audio_feature_map = audio_feature.expand(-1, -1, last_feat_H, last_feat_W) # BF x 1024 x 7 x 7
        x0 = last_feat.mul(audio_feature_map) # BF x 1024 x 7 x 7

        # encode multi-scale visual features
        fpn_feat = self.neck([x1, x2, x3, torch.cat((x4, x0), dim=1)])
        # print(conv1_feat.shape, conv2_feat.shape, conv3_feat.shape, conv4_feat.shape)

        feature_map_list = list(fpn_feat)
        a_fea_list = [None] * 4

        # if len(self.tpavi_stages) > 0:
        #     if (not self.tpavi_vv_flag) and (not self.tpavi_va_flag):
        #         raise Exception('tpavi_vv_flag and tpavi_va_flag cannot be False at the same time if len(tpavi_stages)>0, \
        #             tpavi_vv_flag is for video self-attention while tpavi_va_flag indicates the standard version (audio-visual attention)')
        #     for i in self.tpavi_stages:
        #         tpavi_count = 0
        #         conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
        #         if self.tpavi_vv_flag:
        #             conv_feat_vv = self.tpavi_vv(feature_map_list[i], stage=i)
        #             conv_feat += conv_feat_vv
        #             tpavi_count += 1
        #         if self.tpavi_va_flag:
        #             conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], audio_feature, stage=i)
        #             conv_feat += conv_feat_va
        #             tpavi_count += 1
        #             a_fea_list[i] = a_fea
        #         conv_feat /= tpavi_count
        #         feature_map_list[i] = conv_feat # update features of stage-i which conduct TPAVI

        pred = self.head(feature_map_list)

        return pred, feature_map_list, a_fea_list


    def initialize_audioclip_weights(self,):
        self.aclp.load_state_dict(torch.load(self.audioclip_path, map_location='cpu'), strict=False)
        print(f'==> Load audioclip parameters pretrained on Audioset from {self.audioclip_path}')


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    model = Pred_endecoder(channel=256, tpavi_stages=[0,1,2,3], tpavi_va_flag=True)
    output = model(imgs)
    pdb.set_trace()