import torch
import torch.nn as nn
import torchvision.models as models
from .audioclip import AudioCLIP
from model.TPAVI import TPAVIModule
import pdb
import numpy as np
import torch.nn.functional as F
from torchvggish import vggish
from config import cfg

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

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
    def __init__(self, channel=256, args=None, config=None, tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True,
                 audioclip_path=None):
        super(Pred_endecoder, self).__init__()
        self.cfg = config
        self.args = args
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag

        self.audioclip_path = audioclip_path

        self.aclp = AudioCLIP(pretrained=None)
        self.visual_encoder = self.aclp.visual
        visual_weights = torch.load(self.audioclip_path)
        import collections
        visual_params = collections.OrderedDict()
        if "CLIP" in audioclip_path:
            for k, v in visual_weights.items():
                if k.startswith('visual'):
                    visual_params[k.replace('visual.', '')] = v
        elif "resnet" in audioclip_path:
            for k, v in visual_weights.items():
                if k.startswith('conv1') or k.startswith('bn1'):
                    continue
                visual_params[k.replace('visual.', '')] = v
        else:
            raise NotImplementedError
        self.visual_encoder.load_state_dict(visual_params, strict=False)
        self.audio_encoder = audio_extractor(self.cfg, 'cpu')
        self.audio_encoder.eval()
        self.audio_projector = nn.Linear(128, 1024)
        self.relu = nn.ReLU(inplace=True)

        if args.fusion == 'concate' or args.fusion == 'normed_concate':
            self.neck = FPN_Neck([256, 512, 1024, 3072], 256, 0, 4, up_sample_mode=args.fpn_upmode)
        elif args.fusion == 'naive_concate':
            self.neck = FPN_Neck([256, 512, 1024, 4096], 256, 0, 4, up_sample_mode=args.fpn_upmode)
        elif args.fusion == 'score_fusion':
            self.neck = FPN_Neck([256, 512, 1024, 2049], 256, 0, 4, up_sample_mode=args.fpn_upmode)
        elif args.fusion == 'skip':
            self.neck = FPN_Neck([256, 512, 1024, 1024], 256, 0, 4, up_sample_mode=args.fpn_upmode)
        elif args.fusion == 'noprior':
            self.neck = FPN_Neck([256, 512, 1024, 3072], 256, 0, 4, up_sample_mode=args.fpn_upmode)
        elif args.fusion == 'none':
            self.neck = FPN_Neck([256, 512, 1024, 2048], 256, 0, 4, up_sample_mode=args.fpn_upmode)
        else:
            raise NotImplementedError('Not implemented fusion method: {}'.format(args.fusion))
        self.head = FPN_Head([4, 8, 16, 32], [256, 256, 256, 256], 128, graph_size=32, class_size=1)


    def forward(self, x, audio=None):
        self.audio_encoder.eval()
        [x1, x2, x3, x4, last_feat], x_final = self.visual_encoder(x)
        audio_feature = self.audio_encoder(audio) # BF x 128
        audio_feature = self.audio_projector(audio_feature) # BF x 1024
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
        if self.args.fusion == 'normed_concate' or self.args.fusion == 'score_fusion':
            audio_feature = F.normalize(audio_feature, dim=1)
            last_feat = F.normalize(last_feat, dim=1)
            audio_feature_map = audio_feature.expand(-1, -1, last_feat_H, last_feat_W) # BF x 1024 x 7 x 7
            x0 = last_feat.mul(audio_feature_map) # BF x 1024 x 7 x 7
            score_map = x0.sum(dim=1, keepdim=True) # BF x 1 x 7 x 7
        else:
            audio_feature_map = audio_feature.expand(-1, -1, last_feat_H, last_feat_W) # BF x 1024 x 7 x 7
            x0 = last_feat.mul(audio_feature_map) # BF x 1024 x 7 x 7

        # encode multi-scale visual features
        if self.args.fusion == 'concate' or self.args.fusion == 'normed_concate':
            fpn_feat = self.neck([x1, x2, x3, torch.cat((x4, x0), dim=1)])
        elif self.args.fusion == 'naive_concate':
            fpn_feat = self.neck([x1, x2, x3, torch.cat((x4, last_feat, audio_feature_map), dim=1)])
        elif self.args.fusion == 'score_fusion':
            fpn_feat = self.neck([x1, x2, x3, torch.cat((x4, score_map), dim=1)])
        elif self.args.fusion == 'skip':
            fpn_feat = self.neck([x1, x2, x3, x0])
        elif self.args.fusion == 'noprior':
            fpn_feat = self.neck([x1, x2, x3, torch.cat((x4, audio_feature_map), dim=1)])
        elif self.args.fusion == 'none':
            fpn_feat = self.neck([x1, x2, x3, x4])
        else:
            raise NotImplementedError('Not implemented fusion method: {}'.format(self.args.fusion))
        
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