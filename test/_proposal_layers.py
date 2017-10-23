import tensorflow as tf
import numpy as np
from layer_utils.generate_anchors import generate_anchors
from model.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps

BEFORE_NMS_TOP = 600
AFTER_NMS_TOP = 300


def bbox_target_to_region(rpn_bbox_pred, anchors):
    width = anchors[:, 2] - anchors[:, 0] + 1
    height = anchors[:, 3] - anchors[:, 1] + 1
    ctr_x = anchors[:, 0] + 0.5*width
    ctr_y = anchors[:, 1] + 0.5*height

    dx = anchors[:, 0::4]
    dy = anchors[:, 1::4]
    dw = anchors[:, 2::4]
    dh = anchors[:, 3::4]

    roi_x_ctr = dx*width[:, np.newaxis] + ctr_x[:, np.newaxis]
    roi_y_ctr = dy*height[:, np.newaxis] + ctr_y[:, np.newaxis]
    roi_x_width = np.exp(dw)*width[:, np.newaxis]
    roi_y_height = np.exp(dh)*height[:, np.newaxis]

    rois = np.zeros_like(rpn_bbox_pred)
    rois[:, 0::4] = roi_x_ctr - 0.5*roi_x_width
    rois[:, 1::4] = roi_y_ctr - 0.5*roi_y_height
    rois[:, 2::4] = roi_x_ctr + 0.5*roi_x_width
    rois[:, 3::4] = roi_y_ctr + 0.5*roi_y_height
    return rois


def bbox_transform(ex_rois, gt_rois):
    width = ex_rois[:, 2::4] - ex_rois[:, 0::4] + 1.0
    height = ex_rois[:, 3::4] - ex_rois[:, 1::4] + 1.0
    ctr_x = ex_rois[:, 0::4] + 0.5*width
    ctr_y = ex_rois[:, 1::4] + 0.5*height

    gt_width = gt_rois[:, 2::4] - gt_rois[:, 0::4] + 1.0
    gt_height = gt_rois[:, 3::4] - gt_rois[:, 1::4] + 1.0
    gt_ctr_x = gt_rois[:, 0::4] + 0.5*gt_width
    gt_ctr_y = gt_rois[:, 1::4] + 0.5*gt_height

    target_boxs = np.zeros_like(ex_rois)
    target_boxs[:, 0::4] = (gt_ctr_x - ctr_x)/width
    target_boxs[:, 1::4] = (gt_ctr_y - ctr_y)/height
    target_boxs[:, 2::4] = np.log(np.abs(gt_width/width))
    target_boxs[:, 3::4] = np.log(np.abs(gt_height/height))

    return target_boxs


def _generate_rois(rpn_cls_prob, rpn_bbox_pred, anchors, num_anchors):
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    scores = scores.reshape((-1, 1))

    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    proposal = bbox_target_to_region(rpn_bbox_pred, anchors)
    proposal = clip_box(proposal, 600, 1000)

    # before nms
    orders = np.argsort(scores.ravel())[::-1]
    before_nms_indexs = orders[:BEFORE_NMS_TOP]
    scores = scores[before_nms_indexs]
    proposal = proposal[before_nms_indexs]

    proposal = proposal.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)

    print(proposal.shape)
    print(scores.shape)
    # after nms
    keep = nms(np.hstack((proposal, scores)), 0.7)
    proposal = proposal[keep[:AFTER_NMS_TOP], :]
    scores = scores[keep[:AFTER_NMS_TOP], :]
    indexs = np.zeros((proposal.shape[0], 1), dtype=np.float32)
    proposal = np.hstack((indexs, proposal.astype(np.float32, copy=False)))
    return proposal, scores


def _anchor_target_layer(anchors, gt_boxes, im_info, feat_stride, num_anchors, rpn_cls_score):
    height, width = rpn_cls_score.shape[1:3]
    indexs = np.where((anchors[:, 0] > 0) &
                      (anchors[:, 1] > 0) &
                      (anchors[:, 2] < width*feat_stride) &
                      (anchors[:, 3] < height*feat_stride))[0]
    inside_anchors = anchors[indexs]

    labels = np.zeros((len(indexs),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(inside_anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    arg_max_overlaps = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), arg_max_overlaps]
    gt_arg_max_overlaps = np.argmax(overlaps, axis=0)
    gt_max_overlaps = overlaps[gt_arg_max_overlaps, np.arange(overlaps.shape[1])]

    gt_arg_max_overlaps = np.where(overlaps == gt_max_overlaps)

    labels[max_overlaps < 0.3] = 0
    labels[gt_arg_max_overlaps] = 1
    labels[max_overlaps > 0.7] = 1


def _unmap_label(labels, total_anchors, inds_inside, fill=-1):
    total_labels = np.zeros((total_anchors, ), dtype=np.float32)
    total_labels.fill(fill)
    total_labels[inds_inside] = labels
    return total_labels



def generate_anchors_pre(height, width,
                         aspect_ratio=[0.5, 1.0, 2.0],
                         aspect_scale=[8, 16, 32],
                         feat_stride=16):
    anchors = generate_anchors()
    anchor_nums = anchors.shape[0]
    width = np.arange(width)*feat_stride
    height = np.arange(height)*feat_stride
    shift_x, shift_y = np.meshgrid(width, height)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    print(shifts)
    anchors = anchors.reshape(1, anchor_nums, 4) + shifts.reshape(1, shifts.shape[0], 4).transpose([1, 0, 2])
    anchors = anchors.reshape(shifts.shape[0]*anchor_nums, 4).astype(np.float32)
    print(anchors.shape)
    print(anchors)
    length = anchors.shape[0]
    return anchors, length


def clip_box(boxs, height, width):
    boxs[:, 0::4] = np.maximum(np.minimum(boxs[:, 0::4], width-1), 0)
    boxs[:, 1::4] = np.maximum(np.minimum(boxs[:, 1::4], height-1), 0)
    boxs[:, 2::4] = np.maximum(np.minimum(boxs[:, 2::4], width-1), 0)
    boxs[:, 3::4] = np.maximum(np.minimum(boxs[:, 3::4], height-1), 0)
    return boxs


if __name__ == '__main__':
    # generate_anchors_pre(3, 3)
    # a = np.random.randn(30, 4)
    # b = np.random.randn(30, 4)
    # target = bbox_transform(a, b)
    # print(target.shape)

    nums = 40*60*9
    rpn_box_scores = np.random.randn(1, 40, 60, 18)
    rpn_box_pred = np.random.randn(1, 40, 60, 36)
    anchors = np.random.randn(nums, 4)
    _generate_rois(rpn_box_scores, rpn_box_pred, anchors, 9)


    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    anchors = anchors.reshape((1, 9, 4)) + shifts.reshape((1, shifts.shape[0], 4)).transpose([1, 0, 2])
