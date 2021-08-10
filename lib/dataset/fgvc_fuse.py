# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import copy
import json_tricks as json
from collections import OrderedDict
import cv2
import random
import torch

import numpy as np
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset
# from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

logger = logging.getLogger(__name__)

class FGVCDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, class_transform=None, landmark_transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        # self.joints_weight = 1
        self.joints_weight = np.array([1, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 2], dtype=np.float32).reshape((12, 1))

        self.db = []

        self.num_joints = 12
        self.flip_pairs = [[1, 4], [2, 3], [5, 6], [8, 10]]
        self.data_root = cfg.DATASET.ROOT

        self.class_transform = class_transform
        self.landmark_transform = landmark_transform
        self.target_transform = None
        self.class_rotation = 0

        self.class_splits = cfg.FUSE_MODULE.NAME
        # self.class_splits = 'size'
        if 'all' in cfg.FUSE_MODULE.NAME:
            self.class_splits = 'all'
        elif 'size' in cfg.FUSE_MODULE.NAME:
            self.class_splits = 'size'
        elif 'wing' in cfg.FUSE_MODULE.NAME:
            self.class_splits = 'wing'
        elif 'tail' in cfg.FUSE_MODULE.NAME:
            self.class_splits = 'tail'

        if is_train:
            self.labelfile_path = os.path.join(root, 'Class_keypoints_info_train.npy')
            self.split = 'train'
        else:
            self.labelfile_path = os.path.join(root, 'Class_keypoints_info_test.npy')
            self.split = 'val'
        self.db = self._get_db()
        logger.info('=> load {} samples'.format(len(self.db)))
        ##############################################################################

    def _get_db(self):

        gt_db = np.load(self.labelfile_path, allow_pickle=True).tolist()

        return gt_db

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, index):

        this_sample = self.db[index]

        ##########################################
        # Process Landmark Label
        ##########################################
        image_info = {'image': this_sample['keypoint'][0]['img_name']}
        bbox_info = {'xmin': this_sample['box']['xmin'],
                     'ymin': this_sample['box']['ymin'],
                     'width': str(float(this_sample['box']['xmax']) - float(this_sample['box']['xmin'])),
                     'height': str(float(this_sample['box']['ymax']) - float(this_sample['box']['ymin']))}
        joints_vis = []
        joints = []
        for index in range(self.num_joints):
            temp_point_location = []
            temp_x = this_sample['keypoint'][index]['x']
            temp_y = this_sample['keypoint'][index]['y']
            temp_point_location.append(temp_x)
            temp_point_location.append(temp_y)
            temp_vis = this_sample['keypoint'][index]['visible']
            temp_out = this_sample['keypoint'][index]['outside']
            if temp_out == 1 or temp_vis == 0:
                temp_visible = 0
            else:
                temp_visible = 1
            joints_vis.append(temp_visible)
            joints.append(temp_point_location)

        image_name = image_info['image']
        image_path = os.path.join(self.data_root, 'data/images', image_name)
        r = 0

        joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
        joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
        if self.split != 'test':
            joints = np.array(joints)
            joints[:, 0:2] = joints[:, 0:2] - 1
            joints_vis = np.array(joints_vis)
            assert len(joints) == self.num_joints, \
                'joint num diff: {} vs {}'.format(len(joints),
                                                  self.num_joints)

            joints_3d[:, 0:2] = joints[:, 0:2]
            joints_3d_vis[:, 0] = joints_vis[:]
            joints_3d_vis[:, 1] = joints_vis[:]

        data_numpy = cv2.imread(image_path)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_path))

        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if self.split == 'train':
            sf = self.scale_factor
            sf = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * self.rotation_factor,
                        -self.rotation_factor * 2,
                        self.rotation_factor * 2) if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints_3d, joints_3d_vis = fliplr_joints(
                    joints_3d, joints_3d_vis, data_numpy.shape[1], self.flip_pairs)
        else:
            sf = 1

        trans, trans_inv = FGVC_get_affine_transform(bbox_info=bbox_info, rot=r, sf=sf,
                                                     output_size=self.image_size)
        landmark_input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        # #####################################
        # from PIL import Image
        # im = Image.fromarray(landmark_input)
        # im.save(os.path.join('./visualize', 'landmark_' + image_name))
        # ######################################

        if self.landmark_transform is not None:
            landmark_input = self.landmark_transform(landmark_input)

        ori_joints_3d = joints_3d.copy()

        for i in range(self.num_joints):
            joints_3d[i, 0:2] = affine_transform(joints_3d[i, 0:2], trans)

        landmark_target, landmark_target_weight = self.generate_target(joints_3d, joints_3d_vis)
        landmark_target = torch.from_numpy(landmark_target)
        landmark_target_weight = torch.from_numpy(landmark_target_weight)
        wh_rate = float(bbox_info['width']) / float(bbox_info['height'])

        ##########################################
        # Process Class Label
        ##########################################
        size_target = this_sample['class']['size'] - 1
        wing_target = this_sample['class']['wing'] - 1
        tail_target = this_sample['class']['tail'] - 1

        img_width = np.shape(data_numpy)[1]
        img_height = np.shape(data_numpy)[0]

        if self.class_splits == 'size':
            bbox_info = {'xmin': this_sample['box']['xmin'],
                         'ymin': this_sample['box']['ymin'],
                         'width': str(float(this_sample['box']['xmax']) - float(this_sample['box']['xmin'])),
                         'height': str(float(this_sample['box']['ymax']) - float(this_sample['box']['ymin']))}
        elif self.class_splits == 'wing':
            left_wing_tip = np.array([this_sample['keypoint'][1]['x'], this_sample['keypoint'][1]['y']])
            left_wing_front = np.array([this_sample['keypoint'][2]['x'], this_sample['keypoint'][2]['y']])
            right_wing_front = np.array([this_sample['keypoint'][3]['x'], this_sample['keypoint'][3]['y']])
            right_wing_tip = np.array([this_sample['keypoint'][4]['x'], this_sample['keypoint'][4]['y']])
            right_wing_back = np.array([this_sample['keypoint'][5]['x'], this_sample['keypoint'][5]['y']])
            left_wing_back = np.array([this_sample['keypoint'][6]['x'], this_sample['keypoint'][6]['y']])
            xmin = max(min(left_wing_tip[0], left_wing_front[0], right_wing_front[0],
                           right_wing_tip[0], right_wing_back[0], left_wing_back[0]) -
                       np.linalg.norm(left_wing_front - left_wing_back) / 4, 0)
            ymin = max(min(left_wing_tip[1], left_wing_front[1], right_wing_front[1],
                           right_wing_tip[1], right_wing_back[1], left_wing_back[1]) -
                       np.linalg.norm(left_wing_front - left_wing_back) / 4, 0)
            width = min(max(left_wing_tip[0], left_wing_front[0], right_wing_front[0],
                            right_wing_tip[0], right_wing_back[0], left_wing_back[0]) -
                        min(left_wing_tip[0], left_wing_front[0], right_wing_front[0],
                            right_wing_tip[0], right_wing_back[0], left_wing_back[0]),
                        int(this_sample['box']['xmax']) - float(this_sample['box']['xmin']))
            height = min(max(left_wing_tip[1], left_wing_front[1], right_wing_front[1],
                             right_wing_tip[1], right_wing_back[1], left_wing_back[1]) -
                         min(left_wing_tip[1], left_wing_front[1], right_wing_front[1],
                             right_wing_tip[1], right_wing_back[1], left_wing_back[1]),
                         int(this_sample['box']['ymax']) - float(this_sample['box']['ymin']))
            # width = max(width, height)
            # height = max(width, height)
            width += np.linalg.norm(left_wing_front - left_wing_back) / 2
            height += np.linalg.norm(left_wing_front - left_wing_back) / 2
            if xmin + width > img_width:
                width = img_width - xmin
            if ymin + height > img_height:
                height = img_height - ymin
            bbox_info = {'xmin': xmin,
                         'ymin': ymin,
                         'width': width,
                         'height': height}
        elif self.class_splits == 'tail':
            tail_tip = np.array([this_sample['keypoint'][7]['x'], this_sample['keypoint'][7]['y']])
            tail_left = np.array([this_sample['keypoint'][8]['x'], this_sample['keypoint'][8]['y']])
            tail_up = np.array([this_sample['keypoint'][9]['x'], this_sample['keypoint'][9]['y']])
            tail_right = np.array([this_sample['keypoint'][10]['x'], this_sample['keypoint'][10]['y']])
            tail_front = np.array([this_sample['keypoint'][11]['x'], this_sample['keypoint'][11]['y']])
            xmin = max(min(tail_tip[0], tail_left[0], tail_up[0],
                           tail_right[0], tail_front[0]) -
                       np.linalg.norm(tail_tip - tail_front) / 4, 0)
            ymin = max(min(tail_tip[1], tail_left[1], tail_up[1],
                           tail_right[1], tail_front[1]) -
                       np.linalg.norm(tail_tip - tail_front) / 4, 0)
            width = min(max(tail_tip[0], tail_left[0], tail_up[0],
                            tail_right[0], tail_front[0]) -
                        min(tail_tip[0], tail_left[0], tail_up[0],
                            tail_right[0], tail_front[0]),
                        int(this_sample['box']['xmax']) - float(this_sample['box']['xmin']))
            height = min(max(tail_tip[1], tail_left[1], tail_up[1],
                             tail_right[1], tail_front[1]) -
                         min(tail_tip[1], tail_left[1], tail_up[1],
                             tail_right[1], tail_front[1]),
                         int(this_sample['box']['ymax']) - float(this_sample['box']['ymin']))
            width += np.linalg.norm(tail_tip - tail_front) / 2
            height += np.linalg.norm(tail_tip - tail_front) / 2
            if xmin + width > img_width:
                width = img_width - xmin
            if ymin + height > img_height:
                height = img_height - ymin

            bbox_info = {'xmin': xmin,
                         'ymin': ymin,
                         'width': width,
                         'height': height}
        elif self.class_splits == 'all':
            bbox_info = {'xmin': this_sample['box']['xmin'],
                         'ymin': this_sample['box']['ymin'],
                         'width': str(float(this_sample['box']['xmax']) - float(this_sample['box']['xmin'])),
                         'height': str(float(this_sample['box']['ymax']) - float(this_sample['box']['ymin']))}
        else:
            bbox_info = {'xmin': this_sample['box']['xmin'],
                         'ymin': this_sample['box']['ymin'],
                         'width': str(float(this_sample['box']['xmax']) - float(this_sample['box']['xmin'])),
                         'height': str(float(this_sample['box']['ymax']) - float(this_sample['box']['ymin']))}

        trans, trans_inv = FGVC_get_affine_transform(bbox_info=bbox_info, rot=r, sf=sf,
                                                     output_size=self.image_size)
        class_input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        # #####################################
        # from PIL import Image
        # im = Image.fromarray(class_input)
        # im.save(os.path.join('./visualize/fuse_input', 'class_' + image_name))
        # ######################################

        if self.is_train:
            if self.flip and random.random() <= 0.5:
                class_input = class_input[:, ::-1, :].copy()

        if self.class_transform is not None:
            class_input = self.class_transform(class_input)
        else:
            class_input = class_input

        if self.class_splits == 'size':
            class_target = size_target
            class_tensor = np.zeros((3,)).astype('float32')
            class_tensor[class_target] = 1.0
            class_tensor = torch.from_numpy(class_tensor)
        elif self.class_splits == 'wing':
            class_target = wing_target
            class_tensor = np.zeros((2,)).astype('float32')
            class_tensor[class_target] = 1.0
            class_tensor = torch.from_numpy(class_tensor)
        elif self.class_splits == 'tail':
            class_target = tail_target
            class_tensor = np.zeros((2,)).astype('float32')
            class_tensor[class_target] = 1.0
            class_tensor = torch.from_numpy(class_tensor)
        elif self.class_splits == 'all':
            class_target = {'size': size_target,
                            'tail': tail_target,
                            'wing': wing_target}
            size_class_tensor = np.zeros((3,)).astype('float32')
            size_class_tensor[class_target['size']] = 1.0
            size_class_tensor = torch.from_numpy(size_class_tensor)
            wing_class_tensor = np.zeros((2,)).astype('float32')
            wing_class_tensor[class_target['wing']] = 1.0
            wing_class_tensor = torch.from_numpy(wing_class_tensor)
            tail_class_tensor = np.zeros((2,)).astype('float32')
            tail_class_tensor[class_target['tail']] = 1.0
            tail_class_tensor = torch.from_numpy(tail_class_tensor)
            class_tensor = {'size': size_class_tensor,
                            'tail': tail_class_tensor,
                            'wing': wing_class_tensor}
        else:
            raise RuntimeError('Unrecognized Splits !!!')

        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        ##########################################
        # Fuse input and label
        ##########################################

        if self.split == 'train':
            meta = {
                'image_name': image_name,
                'image_path': image_path,
                'joints': joints,
                'joints_vis': joints_vis,
                'rotation': r,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'wh_rate': wh_rate,
                'size_target': size_target,
                'wing_target': wing_target,
                'tail_target': tail_target,
                'class_tensor': class_tensor,
                'num_joints': self.num_joints,
                'joints_3d_ori': ori_joints_3d
            }
            return class_input, landmark_input, class_target, \
               landmark_target, landmark_target_weight, meta
        else:
            meta = {
                'image_name': image_name,
                'image_path': image_path,
                'joints': joints,
                'joints_vis': joints_vis,
                'rotation': r,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'wh_rate': wh_rate,
                'size_target': size_target,
                'wing_target': wing_target,
                'tail_target': tail_target,
                'class_tensor': class_tensor,
                'affine_transform_matrix_inv': trans_inv,
                'num_joints': self.num_joints,
                'joints_3d_ori': ori_joints_3d
            }
            return class_input, landmark_input, class_target, \
               landmark_target, landmark_target_weight, meta

        #meta = {
        #    'image_name': image_name,
        #    'image_path': image_path,
        #    'joints': joints,
        #    'joints_vis': joints_vis,
        #    'rotation': r,
        #    'joints_3d': joints_3d,
        #    'joints_3d_vis': joints_3d_vis,
        #    'wh_rate': wh_rate,
        #    'size_target': size_target,
        #    'wing_target': wing_target,
        #    'tail_target': tail_target,
        #    'class_tensor': class_tensor
        #}

        #return class_input, landmark_input, class_target, \
        #       landmark_target, landmark_target_weight, meta

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight


def FGVC_get_affine_transform(rot, output_size, bbox_info, sf=1.0,
                              shift=np.array([0, 0], dtype=np.float32),
                              inv=0):

    ori_bbox_xmin = float(bbox_info['xmin'])
    ori_bbox_ymin = float(bbox_info['ymin'])
    ori_bbox_w = float(bbox_info['width'])
    ori_bbox_h = float(bbox_info['height'])

    center = [ori_bbox_xmin + ori_bbox_w / 2, ori_bbox_ymin + ori_bbox_h / 2]

    scaled_bbox_w = sf * ori_bbox_w
    scaled_bbox_h = sf * ori_bbox_h
    scaled_bbox_xmin = center[0] - scaled_bbox_w / 2
    scaled_bbox_ymin = center[1] - scaled_bbox_h / 2

    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, scaled_bbox_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scaled_bbox_w * shift
    src[1, :] = np.array(center) + np.array(src_dir) + scaled_bbox_w * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    # src[2:, :] = [center[0] - bbox_h/2, center[1] - bbox_w/2]
    src[2:, :] = get_3rd_point(src[0, :], src[1, :], width=scaled_bbox_w, height=scaled_bbox_h)
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        trans_inv = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    return trans, trans_inv


def FGVC_get_affine_transformold(rot, output_size, bbox_info, sf,
                              shift=np.array([0, 0], dtype=np.float32),
                              inv=0):

    # bbox_xmin = float(bbox_info['xmin'])
    # bbox_ymin = float(bbox_info['ymin'])
    # bbox_w = float(bbox_info['width'])
    # bbox_h = float(bbox_info['height'])
    #
    # center = [bbox_xmin + bbox_w / 2, bbox_ymin + bbox_h / 2]
    #
    # dst_w = output_size[0]
    # dst_h = output_size[1]
    #
    # rot_rad = np.pi * rot / 180
    # src_dir = get_dir([0, bbox_h * -0.5], rot_rad)
    # dst_dir = np.array([0, dst_w * -0.5], np.float32)
    #
    # src = np.zeros((3, 2), dtype=np.float32)
    # dst = np.zeros((3, 2), dtype=np.float32)
    #
    # src[0, :] = center + bbox_w * shift
    # src[1, :] = np.array(center) + np.array(src_dir) + bbox_w * shift
    # dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    # dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    # # src[2:, :] = [center[0] - bbox_h/2, center[1] - bbox_w/2]
    # src[2:, :] = get_3rd_point(src[0, :], src[1, :], width=bbox_w, height=bbox_h)
    # dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    #
    # if inv:
    #     trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    # else:
    #     trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    #
    # return trans
    ori_bbox_xmin = float(bbox_info['xmin'])
    ori_bbox_ymin = float(bbox_info['ymin'])
    ori_bbox_w = float(bbox_info['width'])
    ori_bbox_h = float(bbox_info['height'])

    center = [ori_bbox_xmin + ori_bbox_w / 2, ori_bbox_ymin + ori_bbox_h / 2]

    scaled_bbox_w = sf * ori_bbox_w
    scaled_bbox_h = sf * ori_bbox_h
    scaled_bbox_xmin = center[0] - scaled_bbox_w / 2
    scaled_bbox_ymin = center[1] - scaled_bbox_h / 2

    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, scaled_bbox_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scaled_bbox_w * shift
    src[1, :] = np.array(center) + np.array(src_dir) + scaled_bbox_w * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    # src[2:, :] = [center[0] - bbox_h/2, center[1] - bbox_w/2]
    src[2:, :] = get_3rd_point(src[0, :], src[1, :], width=scaled_bbox_w, height=scaled_bbox_h)
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b, width=None, height=None):
    if width == height:
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    else:
        direct = a - b
        if int(np.linalg.norm(direct)) - int(height/2) > 2:
            raise RuntimeError('direct caculate error of %s !!!')

        length = width / 2
        direct = direct * length / float(height/2)
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result