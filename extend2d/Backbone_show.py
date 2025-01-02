# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

import sys
from pathlib import  Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from hyper_dl.models.backbones import ResNet
from hyper_dl.models.necks import CustomFPN
"""
img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
"""
model = ResNet(depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1)

x = torch.randn(4, 3, 576, 704)
out = model(x)
print(out[0].shape)#1/16 [1,1024,36,44]
print(out[1].shape)#1/32 [1,2048,18,22]

"""
img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
"""
neck = CustomFPN(in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0])
out_neck = neck(out)#[1,256,36.64]
i = 1