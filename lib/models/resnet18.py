# coding: utf-8
"""

@File    :resnet18.py

@Author  : Yi Li

@E-mail  : li_yi@hust.edu.cn

Created on 2019/12/31

"""

from torchvision.models import resnet50, resnet18
import torch
import os


def get_net(cfg, is_train, **kwargs):
    model = resnet18(pretrained=True)
    num_fc_ftr = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_ftr, cfg.CLASS_MODEL.CLASS_NUM)

    if cfg.CLASS_MODEL.PRETRAINED != '':
        state_dict = torch.load(os.path.join(cfg.CLASS_MODEL.PRETRAINED))
        model.load_state_dict(state_dict)

    return model
