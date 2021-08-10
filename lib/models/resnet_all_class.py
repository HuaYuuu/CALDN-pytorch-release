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


def get_net(cfg, **kwargs):
    size_model = resnet18(pretrained=True)
    num_fc_ftr = size_model.fc.in_features
    size_model.fc = torch.nn.Linear(num_fc_ftr, cfg.CLASS_MODEL.CLASS_NUM)

    wing_model = resnet18(pretrained=True)
    num_fc_ftr = wing_model.fc.in_features
    wing_model.fc = torch.nn.Linear(num_fc_ftr, cfg.CLASS_MODEL.CLASS_NUM2)

    tail_model = resnet18(pretrained=True)
    num_fc_ftr = tail_model.fc.in_features
    tail_model.fc = torch.nn.Linear(num_fc_ftr, cfg.CLASS_MODEL.CLASS_NUM3)

    if cfg.CLASS_MODEL.PRETRAINED != '':
        state_dict = torch.load(os.path.join(cfg.CLASS_MODEL.PRETRAINED))
        size_model.load_state_dict(state_dict)

    if cfg.CLASS_MODEL.PRETRAINED2 != '':
        state_dict = torch.load(os.path.join(cfg.CLASS_MODEL.PRETRAINED2))
        wing_model.load_state_dict(state_dict)

    if cfg.CLASS_MODEL.PRETRAINED3 != '':
        state_dict = torch.load(os.path.join(cfg.CLASS_MODEL.PRETRAINED3))
        tail_model.load_state_dict(state_dict)

    return (size_model, wing_model, tail_model)
