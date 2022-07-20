import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from timm.models.cait import Cait
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import constants as const
from anomaly_map import AnomalyMapGenerator


def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same")
        )
    return subnet_conv


def create_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 0 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3

        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )

    return nodes


class FastFlow(nn.Module):
    def __init__(self, backbone, pretrained, flow_steps, input_size, conv3x3_only=False, hidden_ratio=1.0):
        super().__init__()

        self.input_size = input_size

        if backbone in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone, pretrained=pretrained)
            channels = [768]
            scales = [16]
        elif backbone in [const.BACKBONE_RESNET18, const.BACKBONE_WIDE_RESNET50]:
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=[1, 2, 3]
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformer, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True
                    )
                )
        else:
            raise ValueError(
                f'Backbone {backbone} is not supported. List of available backbones are'
                f'[CaiT, DeiT, ResNet-18, Wide-ResNet50_2]'
            )

        # Feature extractor is not trainable. Only FastFlow block is trainable.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fast_flow_blocks = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.fast_flow_blocks.append(
                create_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps
                )
            )
        
        self.anomaly_map_generator = AnomalyMapGenerator()

    def forward(self, x):
        self.feature_extractor.eval()
        if isinstance(self.feature_extractor, VisionTransformer):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1
                )

            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index=7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index=40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        outputs = []
        log_jacobians = []
        for i, feature in enumerate(features):
            output, log_jacobian = self.fast_flow_blocks[i](feature)
            outputs.append(output)
            log_jacobians.append(log_jacobian)

        ret = (outputs, log_jacobians)

        if not self.training:
            ret = self.anomaly_map_generator(outputs, self.input_size)

        return ret
