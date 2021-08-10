# coding: utf-8
"""

@File    :size_only.py

@Author  : Yi Li

@E-mail  : li_yi@hust.edu.cn

Created on 2019/12/31

"""

from torchvision.models import resnet50, resnet18
import os
import torch.nn as nn
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GroupConv(nn.Module):

    def __init__(self, inplanes, planes, width=64, norm_layer=None):
        super(GroupConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class Fuse_module(nn.Module):
    def __init__(self, cfg, split='size', **kwargs):
        super(Fuse_module, self).__init__()

        output_channel = cfg.FUSE_MODULE.CHANNEL
        if split == 'size':
            self.group_conv = GroupConv(3, 64)
        elif split == 'wing':
            self.group_conv = GroupConv(2, 64)
        elif split == 'tail':
            self.group_conv = GroupConv(2, 64)

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(64, output_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        logger.info('=> init fuse module weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, class_output, landmark_input):

        class_output = torch.softmax(class_output, dim=1)

        class_output = torch.unsqueeze(class_output, dim=2)
        class_output = torch.unsqueeze(class_output, dim=3)
        # broadcasted_class_output = torch.zeros_like(landmark_input)
        broadcasted_class_output = torch.zeros((class_output.shape[0], class_output.shape[1],
                                                landmark_input.shape[2], landmark_input.shape[3])).cuda()
        broadcasted_class_output += class_output

        x = self.group_conv(broadcasted_class_output)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.relu(x)
        global_attention = torch.sigmoid(x)

        y = self.conv1(landmark_input)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y)

        out = y * global_attention.expand_as(y)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # fuse_out = torch.cat((out, landmark_input), dim=1)

        return out


def get_net(cfg, is_train, split='size', **kwargs):
    model = Fuse_module(cfg=cfg, split=split)
    return model
