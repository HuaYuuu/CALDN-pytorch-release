# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import cv2

from core.evaluate import accuracy, accuracy_withclass, accuracy_savewrongpoints, accuracy_withclassandrate
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from PIL import Image, ImageDraw, ImageFont
from utils.transforms import affine_transform


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    gt_locations = []
    pred_locations = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def validate_fgvc(config, val_loader, val_dataset, model, criterion, output_dir,
                  tb_log_dir, writer_dict=None):
    keypoints_name = ['head', 'left_wing_tip', 'left_wing_front', 'right_wing_front',
                      'right_wing_tip', 'right_wing_back', 'left_wing_back', 'tail_tip',
                      'tail_left', 'tail_up', 'tail_right', 'tail_front']
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    NME = AverageMeter()
    acc_point_sum = np.zeros((config.MODEL.NUM_JOINTS,))
    acc_point_num = np.zeros((config.MODEL.NUM_JOINTS,))

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    gt_locations = []
    pred_locations = []
    start_time = time.time()

    with torch.no_grad():
        end = time.time()

        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            # acc_points, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
            #                                           target.cpu().numpy())
            acc_points, avg_acc, cnt, \
            pred, class_sum, class_right, dist_sum, num_dist_cal = accuracy_withclass(output.cpu().numpy(),
                                                              target.cpu().numpy())

            right_wrong_matrix = accuracy_savewrongpoints(output.cpu().numpy(),
                                                          target.cpu().numpy())

            acc_point_sum, acc_point_num = caculate_each_points_acc(acc_point_sum,
                                                                    acc_point_num,
                                                                    class_sum,
                                                                    class_right)
            # acc_point_sum, acc_point_num = caculate_each_points_acc(acc_point_sum, acc_point_num, acc_points)
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            NME.update(dist_sum / num_dist_cal, num_dist_cal)

            # image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

            # # Check output folder
            # output_heat_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                   config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
            #                                   + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' + str(config.MODEL.IMAGE_SIZE[1])
            #                                   + '_' + config.TRAIN.OPTIMIZER + '_lr1e-3', 'heat',
            #                                   config.TEST.MODEL_FILE.split('/')[-1].split('.')[0] + '/')
            # output_visual_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                     config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
            #                                     + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' + str(
            #                                         config.MODEL.IMAGE_SIZE[1])
            #                                     + '_' + config.TRAIN.OPTIMIZER + '_lr1e-3', 'visual',
            #                                     config.TEST.MODEL_FILE.split('/')[-1].split('.')[0] + '/')
            # output_color_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                    config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
            #                                    + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' + str(config.MODEL.IMAGE_SIZE[1])
            #                                    + '_' + config.TRAIN.OPTIMIZER + '_lr1e-3', 'color',
            #                                    config.TEST.MODEL_FILE.split('/')[-1].split('.')[0] + '/')
            # output_gt_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                 config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
            #                                 + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
            #                                 str(config.MODEL.IMAGE_SIZE[1]) + '_' + config.TRAIN.OPTIMIZER
            #                                 + '_lr1e-3', 'gt', config.TEST.MODEL_FILE.split('/')[-1].split('.')[0] + '/')
            # output_wrong_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                    config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
            #                                    + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
            #                                    str(config.MODEL.IMAGE_SIZE[1]) + '_' + config.TRAIN.OPTIMIZER
            #                                    + '_lr1e-3', 'wrong',
            #                                    config.TEST.MODEL_FILE.split('/')[-1].split('.')[0] + '/')
            # output_ori_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                  config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
            #                                  + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
            #                                  str(config.MODEL.IMAGE_SIZE[1]) + '_' + config.TRAIN.OPTIMIZER
            #                                  + '_lr1e-3', 'ori',
            #                                  config.TEST.MODEL_FILE.split('/')[-1].split('.')[0] + '/')
            # output_ori_visible_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                          config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
            #                                          + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
            #                                          str(config.MODEL.IMAGE_SIZE[1]) + '_' + config.TRAIN.OPTIMIZER
            #                                          + '_lr1e-3', 'ori_visible',
            #                                          config.TEST.MODEL_FILE.split('/')[-1].split('.')[0] + '/')
            # output_ori_visible_largepoint_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                                     config.MODEL.PRETRAINED.split('/')[-1].split('-')[
            #                                                         0].split('_')[-1]
            #                                                     + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
            #                                                     str(config.MODEL.IMAGE_SIZE[
            #                                                             1]) + '_' + config.TRAIN.OPTIMIZER
            #                                                     + '_lr1e-3', 'ori_visible_largepoint',
            #                                                     config.TEST.MODEL_FILE.split('/')[-1].split('.')[
            #                                                         0] + '/')
            # output_gtlarge_folder = os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
            #                                      config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
            #                                      + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
            #                                      str(config.MODEL.IMAGE_SIZE[1]) + '_' + config.TRAIN.OPTIMIZER +
            #                                      '_lr1e-3', 'gtlarge',
            #                                      config.TEST.MODEL_FILE.split('/')[-1].split('.')[0] + '/')
            #
            # output_color_folder = './output/fgvc/pose_hrnet/w32_256x256_adam_lr1e-3/color/'
            # output_gt_folder = './output/fgvc/pose_hrnet/w32_256x256_adam_lr1e-3/gt/'
            # check_makedirs(output_heat_folder)
            # check_makedirs(output_color_folder)
            # check_makedirs(output_gt_folder)
            # check_makedirs(output_visual_folder)
            # check_makedirs(output_wrong_folder)
            # check_makedirs(output_ori_folder)
            # check_makedirs(output_ori_visible_folder)
            # check_makedirs(output_ori_visible_largepoint_folder)
            # check_makedirs(output_gtlarge_folder)
            #
            # for index in range(input.shape[0]):
            #     image_ori = cv2.imread(meta['image_path'][index])
            #     image_ori = image_ori[:, :, ::-1]
            #     this_pred_ori = np.zeros_like(pred[index])
            #     for i in range(meta['num_joints'][index]):
            #         temp = pred[index][i, 0:2]
            #         temp *= (config.MODEL.IMAGE_SIZE[0] / config.MODEL.HEATMAP_SIZE[0])
            #         temp = affine_transform(temp, meta['affine_transform_matrix_inv'][index])
            #         this_pred_ori[i, :] = temp
            #     ori_image_large = Image.fromarray(image_ori.astype('uint8'))
            #     draw_ori_image = FGVC_draw_keypoints(ori_image_large.copy(), this_pred_ori, 1,
            #                                          visible=meta['joints_vis'][index])
            #     draw_ori_image.save(os.path.join(output_ori_folder, meta['image_name'][index]))
            #
            #     ori_visible_image_large = Image.fromarray(image_ori.astype('uint8'))
            #     draw_ori_visible_image = FGVC_draw_keypoints_visible(ori_visible_image_large, this_pred_ori, 1,
            #                                                          visible=meta['joints_vis'][index])
            #     draw_ori_visible_image.save(os.path.join(output_ori_visible_folder, meta['image_name'][index]))
            #
            #     ori_visible_image_large_largepoint = Image.fromarray(image_ori.astype('uint8'))
            #     draw_ori_visible_image_large_largepoint = FGVC_draw_keypoints_visible_largepoint(
            #         ori_visible_image_large_largepoint,
            #         this_pred_ori, 1,
            #         visible=meta['joints_vis'][index], color='aqua')
            #     draw_ori_visible_image_large_largepoint.save(
            #         os.path.join(output_ori_visible_largepoint_folder, meta['image_name'][index]))
            #
            #     # Save visualize result
            #     image_temp = Image.fromarray(image_ori.astype('uint8'))
            #     color_draw_image = FGVC_draw_keypoints(image_temp, this_pred_ori, 1, visible=meta['joints_vis'][index])
            #     color_draw_image.save(os.path.join(output_color_folder, meta['image_name'][index]))
            #
            #     # Save gt result
            #     image_temp = Image.fromarray(image_ori.astype('uint8'))
            #     gt_draw_image = FGVC_draw_keypoints(image_temp, meta['joints_3d_ori'][index][:, 0:2].numpy(),
            #                                         1, visible=meta['joints_vis'][index])
            #     gt_draw_image.save(os.path.join(output_gt_folder, meta['image_name'][index]))
            #
            #     # Save gt large result
            #     image_temp = Image.fromarray(image_ori.astype('uint8'))
            #     gt_draw_image = FGVC_draw_keypoints_visible_largepoint(image_temp,
            #                                                            meta['joints_3d_ori'][index][:, 0:2].numpy(),
            #                                                            1, visible=meta['joints_vis'][index],
            #                                                            color='lime')
            #     gt_draw_image.save(os.path.join(output_gtlarge_folder, meta['image_name'][index]))
            #
            #     concate_image((ori_image_large, gt_draw_image, draw_ori_visible_image_large_largepoint),
            #                   os.path.join(output_visual_folder, meta['image_name'][index]))
            #
            #     gt_locations.append(meta['joints_3d_ori'][index][:, 0:2].numpy())
            #     pred_locations.append(this_pred_ori)
            #
            #     for keypoint_index in range(config.MODEL.NUM_JOINTS):
            #         if right_wrong_matrix[keypoint_index][index] < 0.5:
            #             this_point_name = keypoints_name[keypoint_index]
            #             concate_image((ori_image_large, gt_draw_image, draw_ori_visible_image_large_largepoint),
            #                           os.path.join(output_wrong_folder,
            #                                        this_point_name + '_' + meta['image_name'][index]))

            # for index in range(input.shape[0]):
            #
            #     image = torch.squeeze(input[index].detach().cpu())
            #     if image.shape[0] == 3:
            #         pass
            #     elif image.shape[0] == 6:
            #         image = image[0:3]
            #     else:
            #         raise RuntimeError('Tensor Type Error !!!')
            #     image = tensor2im(image)
            #
            #     ori_image = Image.fromarray(image.astype('uint8')).convert('RGB')
            #     ori_image = ori_image.resize((config.MODEL.IMAGE_SIZE[0]*2, config.MODEL.IMAGE_SIZE[1]*2))
            #
            #     # Save visualize result
            #     image_temp = ori_image.copy()
            #     color_draw_image = FGVC_draw_keypoints(image_temp, pred[index], 8, visible=meta['joints_vis'][index])
            #     color_draw_image.save(os.path.join(output_color_folder, meta['image_name'][index]))
            #
            #     # Save gt result
            #     image_temp = ori_image.copy()
            #     gt_draw_image = FGVC_draw_keypoints(image_temp, meta['joints_3d'][index][:, 0:2].numpy(),
            #                                         2, visible=meta['joints_vis'][index])
            #     gt_draw_image.save(os.path.join(output_gt_folder, meta['image_name'][index]))
            #
            #     concate_image((ori_image, gt_draw_image, color_draw_image),
            #                   os.path.join(output_visual_folder, meta['image_name'][index]))
            #
            #     for keypoint_index in range(config.MODEL.NUM_JOINTS):
            #         if right_wrong_matrix[keypoint_index][index] < 0.5:
            #             this_point_name = keypoints_name[keypoint_index]
            #             concate_image((ori_image, gt_draw_image, color_draw_image),
            #                           os.path.join(output_wrong_folder,
            #                                        this_point_name + '_' + meta['image_name'][index]))

        print('Total Test Result: [{}/{}] PCK {acc.avg:.5f} '
              .format(len(val_loader), len(val_loader), acc=acc))
        msg = 'Total Landmark Test Result: [{}/{}] PCK {acc.avg:.5f} ' \
            .format(len(val_loader), len(val_loader), acc=acc)
        logger.info(msg)
        print('Total Test Result: [{}/{}] NME {NME.avg:.5f} '
              .format(len(val_loader), len(val_loader), NME=NME))
        msg = 'Total Test Result: [{}/{}] NME {NME.avg:.5f} ' \
            .format(len(val_loader), len(val_loader), NME=NME)
        logger.info(msg)

        end_time = time.time()

        print('Total Time:  ' + str(end_time-start_time))
        msg = 'Total Time:  ' + str(end_time-start_time)
        logger.info(msg)

        print('Total Test Time: {Time.avg:.5f} '
              .format(Time=batch_time))
        msg = 'Total Test Time: {Time.avg:.5f} '\
              .format(Time=batch_time)
        logger.info(msg)

        # caculate each category result
        acc_each_point = np.divide(acc_point_sum, acc_point_num)
        if 'fgvc' in config.DATASET.DATASET:
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
            with open(os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
                                            config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
                                            + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
                                            str(config.MODEL.IMAGE_SIZE[1]) + '_' + config.TRAIN.OPTIMIZER
                                              + '_lr1e-3', 'results.txt'), 'a+') as result_file:
                result_file.write('\n \n \nTest Model:' + config.TEST.MODEL_FILE.split('/')[-1])
                result_file.write('Each category Results: \n'
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
                result_file.write('Total Test Result: [{}/{}] PCK {acc.avg:.5f} '
                                  .format(len(val_loader), len(val_loader), acc=acc))
                result_file.write('Total Test Result: [{}/{}] NME {NME.avg:.5f} '
                                  .format(len(val_loader), len(val_loader), NME=NME))

            np.save(os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
                                 config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
                                 + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
                                 str(config.MODEL.IMAGE_SIZE[1]) + '_' + config.TRAIN.OPTIMIZER
                                 + '_lr1e-3', 'gt_locations.npy'), gt_locations)
            np.save(os.path.join('./output', config.DATASET.DATASET, config.MODEL.NAME,
                                 config.MODEL.PRETRAINED.split('/')[-1].split('-')[0].split('_')[-1]
                                 + '_' + str(config.MODEL.IMAGE_SIZE[0]) + 'x' +
                                 str(config.MODEL.IMAGE_SIZE[1]) + '_' + config.TRAIN.OPTIMIZER
                                 + '_lr1e-3', 'pred_locations.npy'), pred_locations)

    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def caculate_each_points_acc(acc_point_sum, acc_point_num, class_sum, class_right):
    for i in range(len(acc_point_sum)):
        if class_sum[i] >= 0:
            acc_point_sum[i] += class_right[i]
            acc_point_num[i] += class_sum[i]
    return acc_point_sum, acc_point_num


def FGVC_draw_keypoints(image, locations, zoom, visible=None, keypoint_name=['1', '2', '3', '4',
                                                                             '5', '6', '7', '8',
                                                                             '9', '10', '11', '12']):

    # locations = int(locations)
    locations = locations * zoom
    locations = np.squeeze(locations)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('times.ttf', size=20)
    if visible is None:
        visible = np.ones(1, len(locations))

    for index in range(len(locations)):
        if visible[index] > 0.5:
            draw.ellipse(((locations[index][0] - 2), (locations[index][1] - 2),
                          (locations[index][0] + 2), (locations[index][1] + 2)), fill='red')
            draw.text((locations[index][0], locations[index][1]), keypoint_name[index], fill='yellow', font=font)
        elif visible[index] <= 0.5:
            draw.ellipse(((locations[index][0] - 2), (locations[index][1] - 2),
                          (locations[index][0] + 2), (locations[index][1] + 2)), fill='blue')
            draw.text((locations[index][0], locations[index][1]), keypoint_name[index], fill='white', font=font)
        else:
            raise RuntimeError('Invalid Visible variable !!!')

    return image


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485,0.456,0.406] #dataLoader中设置的mean参数
    std = [0.229,0.224,0.225]  #dataLoader中设置的std参数
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): #如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): #反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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


def concate_image(images, save_name):

    images_list = []
    images_size = []
    for image in images:
        images_list.append(image)
        images_size.append(image.size)

    img_size = images_size[0]

    for size in images_size:
        assert size == img_size

    cut_size = img_size
    new_size = (cut_size[0] * 3, cut_size[1])

    new_image = Image.new('RGB', new_size)
    for i in range(len(images_list)):
        image = images_list[i]
        new_image.paste(image, (i * cut_size[0], 0))

    # new_image.show()
    new_image.save(save_name)


def FGVC_draw_keypoints_visible_largepoint(image, locations, zoom, visible=None, color='green'):

    # locations = int(locations)
    locations = locations * zoom
    locations = np.squeeze(locations)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('times.ttf', size=20)
    if visible is None:
        visible = np.ones(1, len(locations))

    for index in range(len(locations)):
        if visible[index] > 0.5:
            draw.ellipse(((locations[index][0] - 8), (locations[index][1] - 8),
                          (locations[index][0] + 8), (locations[index][1] + 8)), fill=color)
        else:
            pass

    return image


def FGVC_draw_keypoints_visible(image, locations, zoom, visible=None, keypoint_name=['1', '2', '3', '4',
                                                                                     '5', '6', '7', '8',
                                                                                     '9', '10', '11', '12']):

    # locations = int(locations)
    locations = locations * zoom
    locations = np.squeeze(locations)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('times.ttf', size=20)
    if visible is None:
        visible = np.ones(1, len(locations))

    for index in range(len(locations)):
        if visible[index] > 0.5:
            draw.ellipse(((locations[index][0] - 2), (locations[index][1] - 2),
                          (locations[index][0] + 2), (locations[index][1] + 2)), fill='red')
            draw.text((locations[index][0], locations[index][1]), keypoint_name[index], fill='yellow', font=font)
        elif visible[index] <= 0.5:
            pass
        else:
            raise RuntimeError('Invalid Visible variable !!!')

    return image

