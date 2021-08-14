# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import logging
import cv2

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from utils.utils import create_logger
from utils.transforms import flip_back
from core.inference import get_max_preds

import dataset
import models
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    class_model = eval('models.' + cfg.CLASS_MODEL.NAME + '.get_net')(
        cfg, is_train=False
    )
    landmark_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        if 'all' in cfg.FUSE_MODULE.NAME:
            (size_class_model, wing_class_model, tail_class_model) = class_model
            size_class_model.load_state_dict(torch.load(cfg.CLASS_MODEL.PRETRAINED))
            wing_class_model.load_state_dict(torch.load(cfg.CLASS_MODEL.PRETRAINED2))
            tail_class_model.load_state_dict(torch.load(cfg.CLASS_MODEL.PRETRAINED3))
            class_model = (size_class_model, wing_class_model, tail_class_model)
        else:
            class_model.load_state_dict(torch.load(cfg.CLASS_MODEL.PRETRAINED))
        landmark_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        class_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(class_model_state_file))
        class_model.load_state_dict(torch.load(class_model_state_file))
        landmark_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(landmark_model_state_file))
        class_model.load_state_dict(torch.load(landmark_model_state_file))

    if 'all' in cfg.FUSE_MODULE.NAME:
        (size_class_model, wing_class_model, tail_class_model) = class_model
        size_class_model = torch.nn.DataParallel(size_class_model, device_ids=cfg.GPUS).cuda()
        wing_class_model = torch.nn.DataParallel(wing_class_model, device_ids=cfg.GPUS).cuda()
        tail_class_model = torch.nn.DataParallel(tail_class_model, device_ids=cfg.GPUS).cuda()
        class_model = (size_class_model, wing_class_model, tail_class_model)
    else:
        class_model = torch.nn.DataParallel(class_model, device_ids=cfg.GPUS).cuda()
    landmark_model = torch.nn.DataParallel(landmark_model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    class_criterion = torch.nn.CrossEntropyLoss()
    landmark_criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        class_transform=transforms.Compose([
            transforms.ToTensor()]),
        landmark_transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate_fgvc(cfg, valid_loader, valid_dataset, class_model, 
                  landmark_model, class_criterion, landmark_criterion)


def validate_fgvc(cfg, val_loader, val_dataset, class_model, 
                  landmark_model, class_criterion, landmark_criterion):

    # keypoints_name = ['head', 'left_wing_tip', 'left_wing_front', 'right_wing_front',
    #                   'right_wing_tip', 'right_wing_back', 'left_wing_back', 'tail_tip',
    #                   'tail_left', 'tail_up', 'tail_right', 'tail_front']

    batch_time = AverageMeter()
    losses = AverageMeter()

    class_acc = AverageMeter()
    landmark_acc = AverageMeter()
    NME = AverageMeter()
    landmark_acc_point_sum = np.zeros((cfg.MODEL.NUM_JOINTS,))
    landmark_acc_point_num = np.zeros((cfg.MODEL.NUM_JOINTS,))

    # switch to evaluate mode
    if 'all' in cfg.FUSE_MODULE.NAME:
        (size_class_model, wing_class_model, tail_class_model) = class_model
        size_class_model.eval()
        wing_class_model.eval()
        tail_class_model.eval()
        class_model = (size_class_model, wing_class_model, tail_class_model)
    else:
        class_model.eval()
    landmark_model.eval()
    
    class_weight = cfg.CLASS_MODEL.WEIGHT
    landmark_weight = cfg.MODEL.WEIGHT

    idx = 0

    starttime = time.time()
    
    with torch.no_grad():

        for i, (class_input, landmark_input, class_target,
                landmark_target, target_weight, meta) in enumerate(val_loader):

            batchstart = time.time()
            # Generate Prediction
            if 'all' in cfg.FUSE_MODULE.NAME:
                (size_class_model, wing_class_model, tail_class_model) = class_model
                size_class_output = size_class_model(class_input)
                wing_class_output = wing_class_model(class_input)
                tail_class_output = tail_class_model(class_input)
                class_output = {'size': size_class_output,
                                'tail': tail_class_output,
                                'wing': wing_class_output}
            else:
                class_output = class_model(class_input)
            landmark_output = landmark_model(class_output, landmark_input)

            if cfg.TEST.FLIP_TEST:  # Flip Test
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                landmark_input_flipped = np.flip(landmark_input.cpu().numpy(), 3).copy()
                landmark_input_flipped = torch.from_numpy(landmark_input_flipped).cuda()
                landmark_output_flipped = landmark_model(meta['class_tensor'],
                                                         landmark_input_flipped)

                # Flip Prediction Back
                landmark_output_flipped = flip_back(landmark_output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                landmark_output_flipped = torch.from_numpy(landmark_output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if cfg.TEST.SHIFT_HEATMAP:
                    landmark_output_flipped[:, :, :, 1:] = \
                        landmark_output_flipped.clone()[:, :, :, 0:-1]

                # Prediction Fusion
                landmark_output = (landmark_output + landmark_output_flipped) * 0.5

            batchend = time.time()

            # Put Labels to GPUs
            landmark_target = landmark_target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            # Compute Class and Landmark Loss
            landmark_loss = landmark_criterion(landmark_output, landmark_target, target_weight)
            if 'all' in cfg.FUSE_MODULE.NAME:
                size_class_target = class_target['size'].cuda(non_blocking=True)
                size_class_loss = class_criterion(class_output['size'], size_class_target)
                wing_class_target = class_target['wing'].cuda(non_blocking=True)
                wing_class_loss = class_criterion(class_output['wing'], wing_class_target)
                tail_class_target = class_target['tail'].cuda(non_blocking=True)
                tail_class_loss = class_criterion(class_output['tail'], tail_class_target)
                loss = class_weight * size_class_loss + class_weight * wing_class_loss + \
                       class_weight * tail_class_loss + landmark_weight * landmark_loss
            else:
                class_target = class_target.cuda(non_blocking=True)
                class_loss = class_criterion(class_output, class_target)
                loss = class_weight * class_loss + landmark_weight * landmark_loss
            num_images = landmark_input.size(0)
            losses.update(loss.item(), num_images)

            # Measure Accuracy
            if 'all' in cfg.FUSE_MODULE.NAME:
                size_class_avg_acc = caculate_class_acc(class_output['size'], size_class_target)
                wing_class_avg_acc = caculate_class_acc(class_output['wing'], wing_class_target)
                tail_class_avg_acc = caculate_class_acc(class_output['tail'], tail_class_target)
                class_avg_acc = (size_class_avg_acc + wing_class_avg_acc + tail_class_avg_acc) / 3
            else:
                class_avg_acc = caculate_class_acc(class_output, class_target)
            landmark_acc_points, landmark_avg_acc, landmark_cnt, \
            landmark_pred, landmark_class_sum, landmark_class_right, dist_sum, num_dist_cal = accuracy_withclass(
                landmark_output.cpu().numpy(),
                landmark_target.cpu().numpy())
            landmark_acc_point_sum, landmark_acc_point_num = caculate_each_points_acc(
                landmark_acc_point_sum,
                landmark_acc_point_num,
                landmark_class_sum,
                landmark_class_right)

            # Update Accuracy of Each instance
            landmark_acc.update(landmark_avg_acc, landmark_cnt)
            class_acc.update(class_avg_acc)
            NME.update(dist_sum / num_dist_cal, num_dist_cal)

            # measure elapsed time
            batch_time.update(batchend - batchstart)

            # Print Test Info
            idx += num_images
            if (i + 1) % cfg.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Class Accuracy {class_acc.val:.4f} ({class_acc.avg:.4f})\t' \
                      'Landmark Accuracy {landmark_acc.val:.3f} ({landmark_acc.avg:.3f})'.format(
                          i + 1, len(val_loader), batch_time=batch_time,
                          loss=losses, class_acc=class_acc, landmark_acc=landmark_acc)
                logger.info(msg)

        # Compute Time Consumption
        endtime = time.time()
        totaltesttime = endtime - starttime
        totaltesttime = totaltesttime
        batchavgtime = batch_time.avg / cfg.TEST.BATCH_SIZE_PER_GPU

        # caculate result
        PCK = np.divide(np.sum(landmark_acc_point_sum), np.sum(landmark_acc_point_num))
        print('Total Network Test time: {totaltesttime:.5f} Batch Time: {batchavgtime:.5f} ' \
              .format(totaltesttime=totaltesttime, batchavgtime=batchavgtime))
        msg = 'Total Network Test time: {totaltesttime:.5f} Batch Time: {batchavgtime:.5f} ' \
            .format(totaltesttime=totaltesttime, batchavgtime=batchavgtime)
        logger.info(msg)
        print('Total Landmark Test Result: [{}/{}] PCK {acc:.5f} '
              .format(len(val_loader), len(val_loader), acc=PCK))
        msg = 'Total Landmark Test Result: [{}/{}] PCK {acc:.5f} ' \
            .format(len(val_loader), len(val_loader), acc=PCK)
        logger.info(msg)
        print('Total Test Result: [{}/{}] NME {NME.avg:.5f} '
              .format(len(val_loader), len(val_loader), NME=NME))
        msg = 'Total Test Result: [{}/{}] NME {NME.avg:.5f} ' \
            .format(len(val_loader), len(val_loader), NME=NME)
        logger.info(msg)
        print('Total Class Test Result: [{}/{}] Class Acc {acc.avg:.5f} '
              .format(len(val_loader), len(val_loader), acc=class_acc))
        msg = 'Total Class Test Result: [{}/{}] Class Acc {acc.avg:.5f} ' \
            .format(len(val_loader), len(val_loader), acc=class_acc)
        logger.info(msg)

        # Caculate Each Category Result and Print
        acc_each_point = np.divide(landmark_acc_point_sum, landmark_acc_point_num)
        if 'fgvc' in cfg.DATASET.DATASET:
            print('Each category Results: \n'
                  'head:             {:.5f} \n'
                  'left_wing_tip:    {:.5f} \n'
                  'left_wing_front:  {:.5f} \n'
                  'right_wing_front: {:.5f} \n'
                  'right_wing_tip:   {:.5f} \n'
                  'right_wing_back:  {:.5f} \n'
                  'left_wing_back:   {:.5f} \n'
                  'tail_tip:         {:.5f} \n'
                  'tail_left:        {:.5f} \n'
                  'tail_up:          {:.5f} \n'
                  'tail_right:       {:.5f} \n'
                  'tail_front:       {:.5f} \n'
                  .format(acc_each_point[0], acc_each_point[1], acc_each_point[2],
                          acc_each_point[3], acc_each_point[4], acc_each_point[5],
                          acc_each_point[6], acc_each_point[7], acc_each_point[8],
                          acc_each_point[9], acc_each_point[10], acc_each_point[11]))
            msg = 'Each category Results: \n'\
                  'head:             {:.5f} \n'\
                  'left_wing_tip:    {:.5f} \n'\
                  'left_wing_front:  {:.5f} \n'\
                  'right_wing_front: {:.5f} \n'\
                  'right_wing_tip:   {:.5f} \n'\
                  'right_wing_back:  {:.5f} \n'\
                  'left_wing_back:   {:.5f} \n'\
                  'tail_tip:         {:.5f} \n'\
                  'tail_left:        {:.5f} \n'\
                  'tail_up:          {:.5f} \n'\
                  'tail_right:       {:.5f} \n'\
                  'tail_front:       {:.5f} \n'\
                  .format(acc_each_point[0], acc_each_point[1], acc_each_point[2],
                          acc_each_point[3], acc_each_point[4], acc_each_point[5],
                          acc_each_point[6], acc_each_point[7], acc_each_point[8],
                          acc_each_point[9], acc_each_point[10], acc_each_point[11])
            logger.info(msg)

    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def caculate_each_points_acc(acc_point_sum, acc_point_num, class_sum, class_right):
    for i in range(len(acc_point_sum)):
        if class_sum[i] >= 0:
            assert class_right[i] >= 0
            assert class_sum[i] >= 0
            acc_point_sum[i] += class_right[i]
            acc_point_num[i] += class_sum[i]
    return acc_point_sum, acc_point_num


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def calc_dists(preds, target, normalize, target_weight):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            # if target[n, c, 0] > 1 and target[n, c, 1] > 1:
            if target_weight is not None:
                if target_weight[n, c] > 0.5:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
            else:
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
    return dists


def dist_acc_withsum(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal, \
               int(np.less(dists[dist_cal], thr).sum() * 1.0), num_dist_cal
    else:
        return -1


def accuracy_withclass(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''

    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, target_weight = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm, target_weight)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    dist_sum = dists[dist_cal].sum()
    nme = dist_sum / num_dist_cal

    class_sum = np.zeros_like(pred)[0, :, 0]
    class_right = np.zeros_like(pred)[0, :, 0]

    for i in range(len(idx)):
        acc[i + 1], temp_right, temp_sum = dist_acc_withsum(dists[idx[i]])
        class_sum[i] = temp_sum
        class_right[i] = temp_right
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred, class_sum, class_right, dist_sum, num_dist_cal


def caculate_class_acc(class_out, class_label):
    correct = 0
    total = 0
    _, predicted = torch.max(class_out.data, 1)
    total += class_label.size(0)
    correct += (predicted == class_label).sum()
    acc = (correct.__float__() / total)
    return acc


if __name__ == '__main__':
    main()
