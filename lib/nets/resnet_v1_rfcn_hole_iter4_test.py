# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen and Chen Wei
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import numpy as np

from nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from model.config import cfg
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib


def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': False,
    'updates_collections': ops.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc


def resnet_v1_block(scope, base_depth, num_units, stride):
      """Helper function for creating a resnet_v1 bottleneck block.

      Args:
        scope: The scope of the block.
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
          All other units have stride=1.

      Returns:
        A resnet_v1 bottleneck block.
      """
      bottleneck = resnet_v1.bottleneck
      return resnet_utils.Block(scope, bottleneck, [{
          'depth': base_depth * 4,
          'depth_bottleneck': base_depth,
          'stride': 1
      }] * (num_units - 1) + [{
          'depth': base_depth * 4,
          'depth_bottleneck': base_depth,
          'stride': stride
      }])


@add_arg_scope
def bottleneck_hole(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=2,
               outputs_collections=None,
               scope=None):
  with variable_scope.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')

    residual = layers.conv2d(
        inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
    residual = layers_lib.conv2d(residual, depth_bottleneck, [3, 3], stride=1, rate=rate, padding='SAME', scope='conv2')
    residual = layers.conv2d(
        residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')

    output = nn_ops.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v1_block_hole(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v1 bottleneck block.
    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.

    Returns:
      A resnet_v1 bottleneck block.
    """
    bottleneck = bottleneck_hole
    return resnet_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


class resnetv1(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.RESNET.MAX_POOL:
        pre_pool_size = cfg.POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self):
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

    return net

  def build_network(self, sess, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    bottleneck = resnet_v1.bottleneck
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      blocks = [
          resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
          resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
          resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
      ]
    elif self._num_layers == 101:
      blocks = [
          resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
          resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
          resnet_v1_block_hole('block4', base_depth=512, num_units=3, stride=1),
      ]
    elif self._num_layers == 152:
      blocks = [
          resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
          resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
          resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
      ]
    else:
      # other numbers are not supported
      raise NotImplementedError

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS == 3:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    elif cfg.RESNET.FIXED_BLOCKS > 0:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net, _ = resnet_v1.resnet_v1(net,
                                     blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope=self._resnet_scope)

      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[cfg.RESNET.FIXED_BLOCKS:],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    else:  # cfg.RESNET.FIXED_BLOCKS == 0
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)

    self._act_summaries.append(net_conv4)
    self._layers['head'] = net_conv4
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()

      # rpn
      rpn = slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      rpn_cls_score_shape = tf.shape(rpn_cls_score)
      rpn_cls_score_reshape = tf.reshape(rpn_cls_score, shape=[rpn_cls_score_shape[0], rpn_cls_score_shape[1],
                                                               rpn_cls_score_shape[2]*self._num_anchors, 2])
      rpn_cls_prob = tf.nn.softmax(rpn_cls_score_reshape)
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer_bbox,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

      # rfcn   a 1024 1*1 conv layer
      rfcn_net = slim.conv2d(net_conv4, 1024, [1, 1], padding='SAME',
                             weights_initializer=tf.random_normal_initializer(stddev=0.01),
                             weights_regularizer=slim.l2_regularizer(scale=0.0005), scope='refined_reduce_depth',
                             activation_fn=tf.nn.relu)
      # generate k*k*(C+1) score maps
      rfcn_net_classes = slim.conv2d(rfcn_net, cfg.K*cfg.K*(20+1), [1, 1], weights_initializer=tf.random_normal_initializer(stddev=0.01),
                             weights_regularizer=slim.l2_regularizer(scale=0.0005), scope='refined_classes',
                             activation_fn=None)
      rfcn_net_bbox = slim.conv2d(rfcn_net, cfg.K*cfg.K*4*21, [1, 1], weights_regularizer=slim.l2_regularizer(scale=0.0005),
                                  weights_initializer=tf.random_normal_initializer(stddev=0.01), scope='refined_bbox',
                                  activation_fn=None)

      box_ind, bbox = self._normalize_bbox(net_conv4, rois, name='rois2bbox')
      # rfcn pooling layer
      position_sensitive_boxes = []
      ymin, xmin, ymax, xmax = tf.unstack(bbox, axis=1)
      step_y = (ymax - ymin) / cfg.K
      step_x = (xmax - xmin) / cfg.K
      for bin_y in range(cfg.K):
          for bin_x in range(cfg.K):
              box_coordinates = [ymin+bin_y*step_y,
                                 xmin+bin_x*step_x,
                                 ymin+(bin_y+1)*step_y,
                                 xmin+(bin_x+1)*step_x]
              position_sensitive_boxes.append(tf.stack(box_coordinates, axis=1))

      # class with background
      feature_class_split = tf.split(rfcn_net_classes, num_or_size_splits=9, axis=3)

      image_crops = []
      for (split, box) in zip(feature_class_split, position_sensitive_boxes):
          crop = tf.image.crop_and_resize(split, box, tf.to_int32(box_ind), [6, 6])
          image_crops.append(crop)
      position_sensitive_features = tf.add_n(image_crops)/len(image_crops)
      position_sensitive_classes = tf.reduce_mean(position_sensitive_features, axis=[1, 2])
      cls_prob = tf.nn.softmax(position_sensitive_classes)

      # bounding box features
      bbox_target_crops = []
      feature_bbox_split = tf.split(rfcn_net_bbox, num_or_size_splits=9, axis=3)
      for (split, box) in zip(feature_bbox_split, position_sensitive_boxes):
          crop = tf.image.crop_and_resize(split, box, tf.to_int32(box_ind), [6, 6])
          bbox_target_crops.append(crop)
      position_sensitive_bbox_feature = tf.add_n(bbox_target_crops)/len(bbox_target_crops)
      position_sensitive_bbox_feature = tf.reduce_mean(position_sensitive_bbox_feature, axis=[1, 2])

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["cls_score"] = position_sensitive_classes
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = position_sensitive_bbox_feature
    self._predictions["rois"] = rois

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, position_sensitive_bbox_feature

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                           tf.reverse(conv1_rgb, [2])))

  def _normalize_bbox(self, bottom, bbox, name):
      with tf.variable_scope(name_or_scope=name):
          bottom_shape = tf.shape(bottom)
          height = (tf.to_float(bottom_shape[1]) - 1.)*self._feat_stride[0]
          width = (tf.to_float(bottom_shape[2]) - 1)*self._feat_stride[0]

          indexes, x1, y1, x2, y2 = tf.unstack(bbox, axis=1)
          x1 = x1 / width
          y1 = y1 / height
          x2 = x2 / width
          y2 = y2 / height
          # bboxes = tf.stack([y1, x1, y2, x2], axis=1)
          bboxes = tf.stop_gradient(tf.stack([y1, x1, y2, x2], 1))
          return indexes, bboxes