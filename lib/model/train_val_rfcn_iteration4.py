# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """
  def __init__(self, sess, network, rfcn_network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network
    self.rfcn_network = rfcn_network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, sess, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(self.output_dir, filename)

    self.saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indeces of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indeces of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)
    return filename, nfilename

  def get_variables_in_checkpoint_file(self, file_name):
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")

  def train_model(self, sess, max_iters):
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Determine different scales for anchors, see paper
    with sess.graph.as_default():
      # Set the random seed for tensorflow
      tf.set_random_seed(cfg.RNG_SEED)
      # Build the main computation graph
      rpn_layers, proposal_targets = self.net.create_architecture(sess, 'TRAIN', self.imdb.num_classes, scope='rpn_network', tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
      rfcn_layers, _ = self.rfcn_network.create_architecture(sess, 'TRAIN', self.imdb.num_classes, scope='rfcn_network', tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS,
                                            input_rois=rpn_layers['rois'],
                                            roi_scores=rpn_layers['roi_scores'],
                                            proposal_targets=proposal_targets)

      # Define the loss
      rpn_loss = rpn_layers['rpn_loss']
      rfcn_loss = rfcn_layers['rfcn_loss']
      rpn_rfcn_loss = rpn_layers['rfcn_loss']
      # Set learning rate and momentum
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
      momentum = cfg.TRAIN.MOMENTUM
      self.optimizer = tf.train.MomentumOptimizer(lr, momentum)

      # Compute the gradients wrt the loss
      rpn_trainable_variables_stage1 = self.net.get_train_variables('rpn_network')
      rfcn_trainable_variables_stage2 = self.rfcn_network.get_train_variables('rfcn_network')
      rpn_trainable_variables_stage3 = self.net.get_train_variables_stage3('rpn_network')
      rpn_trainable_variables_stage4 = self.net.get_train_variables_stage4('rpn_network')
      gvs_rpn_stage1 = self.optimizer.compute_gradients(rpn_loss, rpn_trainable_variables_stage1)
      gvs_rfcn_stage2 = self.optimizer.compute_gradients(rfcn_loss, rfcn_trainable_variables_stage2)
      gvs_rpn_stage3 = self.optimizer.compute_gradients(rpn_loss, rpn_trainable_variables_stage3)
      gvs_rfcn_stage4 = self.optimizer.compute_gradients(rpn_rfcn_loss, rpn_trainable_variables_stage4)

      train_op_stage1 = self.optimizer.apply_gradients(gvs_rpn_stage1)
      train_op_stage2 = self.optimizer.apply_gradients(gvs_rfcn_stage2)
      train_op_stage3 = self.optimizer.apply_gradients(gvs_rpn_stage3)
      train_op_stage4 = self.optimizer.apply_gradients(gvs_rfcn_stage4)

      # We will handle the snapshots ourselves
      self.saver = tf.train.Saver(max_to_keep=1000000)
      # Write the train and validation information to tensorboard
      self.writer_stage1 = tf.summary.FileWriter(self.tbdir+'/stage1', sess.graph)
      self.writer_stage2 = tf.summary.FileWriter(self.tbdir+'/stage2', sess.graph)
      self.writer_stage3 = tf.summary.FileWriter(self.tbdir+'/stage3', sess.graph)
      self.writer_stage4 = tf.summary.FileWriter(self.tbdir+'/stage4', sess.graph)
      self.valwriter_stage1 = tf.summary.FileWriter(self.tbvaldir+'/stage1')
      self.valwriter_stage2 = tf.summary.FileWriter(self.tbvaldir+'/stage2')
      self.valwriter_stage3 = tf.summary.FileWriter(self.tbvaldir+'/stage3')
      self.valwriter_stage4 = tf.summary.FileWriter(self.tbvaldir+'/stage4')

    # Find previous snapshots if there is any to restore from
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    redstr = '_iter_{:d}.'.format(cfg.TRAIN.STEPSIZE+1)
    sfiles = [ss.replace('.meta', '') for ss in sfiles]
    sfiles = [ss for ss in sfiles if redstr not in ss]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    nfiles = [nn for nn in nfiles if redstr not in nn]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    np_paths = nfiles
    ss_paths = sfiles

    if lsf == 0:
      # Fresh train directly from ImageNet weights
      print('Loading initial model weights from {:s}'.format(self.pretrained_model))
      variables = tf.global_variables()
      # Initialize all variables first
      sess.run(tf.variables_initializer(variables, name='init'))
      var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
      # Get the variables to restore, ignorizing the variables to fix
      variables_to_restore_rpn, variables_to_restore_rfcn = self.net.get_variables_to_restore(variables, var_keep_dic)

      self.saver_rpn = tf.train.Saver(variables_to_restore_rpn)
      self.saver_rpn.restore(sess, self.pretrained_model)

      self.saver_rfcn = tf.train.Saver(variables_to_restore_rfcn)
      self.saver_rfcn.restore(sess, self.pretrained_model)

      print('Loaded.')
      # Need to fix the variables before loading, so that the RGB weights are changed to BGR
      # For VGG16 it also changes the convolutional weights fc6 and fc7 to
      # fully connected weights
      self.net.fix_variables(sess, self.pretrained_model)
      print('Fixed.')
      sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))
      last_snapshot_iter = 0
    else:
      # Get the most recent snapshot and restore
      ss_paths = [ss_paths[-1]]
      np_paths = [np_paths[-1]]

      print('Restorining model snapshots from {:s}'.format(sfiles[-1]))
      self.saver.restore(sess, str(sfiles[-1]))
      print('Restored.')
      # Needs to restore the other hyperparameters/states for training, (TODO xinlei) I have
      # tried my best to find the random states so that it can be recovered exactly
      # However the Tensorflow state is currently not available
      with open(str(nfiles[-1]), 'rb') as fid:
        st0 = pickle.load(fid)
        cur = pickle.load(fid)
        perm = pickle.load(fid)
        cur_val = pickle.load(fid)
        perm_val = pickle.load(fid)
        last_snapshot_iter = pickle.load(fid)

        np.random.set_state(st0)
        self.data_layer._cur = cur
        self.data_layer._perm = perm
        self.data_layer_val._cur = cur_val
        self.data_layer_val._perm = perm_val

        # Set the learning rate, only reduce once
        if last_snapshot_iter > cfg.TRAIN.STEPSIZE:
          sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
        else:
          sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))

    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    stage_infor = ''
    # while iter < max_iters + 1:
    while iter < 200001:
    # while iter < 201:
      # Learning rate
      if iter == 80001:
      # if iter == 81:
        # Add snapshot here before reducing the learning rate
        self.snapshot(sess, iter)
        sess.run(tf.assign(lr, 0.001))
      # Get training data, one batch at a time
      blobs = self.data_layer.forward()
      # stage 1  training rpn layers and backbones  in rpn network
      if iter < 80001:
      # if iter < 81:
        stage_infor = 'stage1'
        if iter == 60001:
        # if iter == 61:
          sess.run(tf.assign(lr, 0.0001))
        timer.tic()
        now = time.time()
        if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
          # Compute the graph with summary
          rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
            self.net.train_rpn_step_with_summary(sess, blobs, train_op_stage1)
          self.writer_stage1.add_summary(summary, float(iter))
          # Also check the summary on the validation set
          blobs_val = self.data_layer_val.forward()
          summary_val = self.net.get_summary(sess, blobs_val)
          self.valwriter_stage1.add_summary(summary_val, float(iter))
          last_summary_time = now
        else:
          # Compute the graph without summary
          rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
            self.net.train_rpn_step(sess, blobs, train_op_stage1)
        timer.toc()
        if iter == 80000:
          self.writer_stage1.close()
          self.valwriter_stage1.close()
      # stage 2 training rfcn layers and backbones  in rfcn network
      elif 80001 <= iter < 200001:
      # elif 81 <= iter < 201:
        stage_infor = 'stage2'
        rpn_loss_cls = 0
        rpn_loss_box = 0
        if iter == 160001:
        # if iter == 161:
          self.snapshot(sess, iter)
          sess.run(tf.assign(lr, 0.0001))
        timer.tic()
        now = time.time()
        if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
          # Compute the graph with summary
          loss_cls, loss_box, total_loss, summary = \
            self.rfcn_network.train_rfcn_step_with_summary_stage2(sess, blobs, train_op_stage2, self.net)
          self.writer_stage2.add_summary(summary, float(iter))
          # Also check the summary on the validation set
          blobs_val = self.data_layer_val.forward()
          summary_val = self.rfcn_network.get_summary_stage2(sess, blobs_val, self.net)
          self.valwriter_stage2.add_summary(summary_val, float(iter))
          last_summary_time = now
        else:
          # Compute the graph without summary
          loss_cls, loss_box, total_loss = \
            self.rfcn_network.train_rfcn_step_stage2(sess, blobs, train_op_stage2, self.net)
        timer.toc()
        if iter == 200000:
          self.writer_stage2.close()
          self.valwriter_stage2.close()
      else:
        raise ValueError('illeagle input iter value')

      # Display training information
      # if iter % (cfg.TRAIN.DISPLAY) == 0:
      if iter % 20 == 0:
        print('iter: %d / %d, stage: %s, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
              '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
              (iter, max_iters, stage_infor, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
        print('speed: {:.3f}s / iter'.format(timer.average_time))

      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        snapshot_path, np_path = self.snapshot(sess, iter)
        np_paths.append(np_path)
        ss_paths.append(snapshot_path)
      iter += 1

    if iter <= 200001:
    # if iter <= 201:
      ###############################################
      #####  merge rfcn_network to rpn_network  #####
      ###############################################
      merged_ops = merged_networks_rfcn2rpn('rfcn_network', 'rpn_network')
      with tf.variable_scope('rpn_network', reuse=True):
        rpn_conv1 = tf.get_variable('resnet_v1_101/block2/unit_1/bottleneck_v1/shortcut/weights')
        rpn_conv2 = tf.get_variable('resnet_v1_101/refined_reduce_depth/weights')
        rpn_conv3 = tf.get_variable('resnet_v1_101/block4/unit_3/bottleneck_v1/conv3/weights')
        rpn_conv4 = tf.get_variable('resnet_v1_101/block3/unit_18/bottleneck_v1/conv2/weights')
        rpn_conv5 = tf.get_variable('resnet_v1_101/block3/unit_14/bottleneck_v1/conv3/weights')
      with tf.variable_scope('rfcn_network', reuse=True):
        rfcn_conv1 = tf.get_variable('resnet_v1_101/block2/unit_1/bottleneck_v1/shortcut/weights')
        rfcn_conv2 = tf.get_variable('resnet_v1_101/refined_reduce_depth/weights')
        rfcn_conv3 = tf.get_variable('resnet_v1_101/block4/unit_3/bottleneck_v1/conv3/weights')
        rfcn_conv4 = tf.get_variable('resnet_v1_101/block3/unit_18/bottleneck_v1/conv2/weights')
        rfcn_conv5 = tf.get_variable('resnet_v1_101/block3/unit_14/bottleneck_v1/conv3/weights')
      with tf.control_dependencies(merged_ops):
        rpn_conv1 = tf.identity(rpn_conv1)
        bool1 = tf.equal(rpn_conv1, rfcn_conv1)
        bool2 = tf.equal(rpn_conv2, rfcn_conv2)
        bool3 = tf.equal(rpn_conv3, rfcn_conv3)
        bool4 = tf.equal(rpn_conv4, rfcn_conv4)
        bool5 = tf.equal(rpn_conv5, rfcn_conv5)
        bool1_val, bool2_val, bool3_val, bool4_val, bool5_val = \
          sess.run([bool1, bool2, bool3, bool4, bool5])
    # stage 3 and stage 4
    sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))
    # while iter < max_iters + 1:
    while iter < 480001:
    # while iter < 401:
        if iter == 280001:
        # if iter == 261:
          # Add snapshot here before reducing the learning rate
          self.snapshot(sess, iter)
          sess.run(tf.assign(lr, 0.001))
        if iter == 400001:
          self.snapshot(sess, iter)
          sess.run(tf.assign(lr, 0.001))

        blobs = self.data_layer.forward()
        # stage 3 training rpn layers only  in rpn network rpn layers
        if 200001 <= iter < 280001:
        # if 201 <= iter < 581:
          stage_infor = 'stage3'
          # if iter == 260001:
          if iter == 260001:
            self.snapshot(sess, iter)
            sess.run(tf.assign(lr, 0.0001))
          timer.tic()
          now = time.time()
          if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
            # Compute the graph with summary
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
              self.net.train_rpn_step_with_summary(sess, blobs, train_op_stage3)
            self.writer_stage3.add_summary(summary, float(iter))
            # Also check the summary on the validation set
            blobs_val = self.data_layer_val.forward()
            summary_val = self.net.get_summary(sess, blobs_val)
            self.valwriter_stage3.add_summary(summary_val, float(iter))
            last_summary_time = now
          else:
            # Compute the graph without summary
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
              self.net.train_rpn_step(sess, blobs, train_op_stage3)
          timer.toc()
          if iter == 280000:
            self.writer_stage3.close()
            self.valwriter_stage3.close()
        # stage 4 training rfcn layer only in rpn network rfcn layers
        elif 280001 <= iter < 400001:
        # elif 581 <= iter < 1401:
          stage_infor = 'stage4'
          if iter == 360001:
          # if iter == 1361:
            sess.run(tf.assign(lr, 0.0001))
          timer.tic()
          now = time.time()
          if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
            # Compute the graph with summary
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
              self.net.train_rpn_step_with_summary(sess, blobs, train_op_stage4)
            self.writer_stage4.add_summary(summary, float(iter))
            # Also check the summary on the validation set
            blobs_val = self.data_layer_val.forward()
            summary_val = self.net.get_summary(sess, blobs_val)
            self.valwriter_stage4.add_summary(summary_val, float(iter))
            last_summary_time = now
          else:
            # Compute the graph without summary
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
              self.net.train_rpn_step(sess, blobs, train_op_stage4)
          timer.toc()
          if iter == 400000:
            self.writer_stage4.close()
            self.valwriter_stage4.close()
        elif 400001 <= iter < 480001:
          # if 401 <= iter < 481:
          stage_infor = 'stage5'
          # if iter == 461:
          if iter == 460001:
            self.snapshot(sess, iter)
            sess.run(tf.assign(lr, 0.0001))
          timer.tic()
          now = time.time()
          if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
            # Compute the graph with summary
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
              self.net.train_rpn_step_with_summary(sess, blobs, train_op_stage3)
            self.writer_stage3.add_summary(summary, float(iter))
            # Also check the summary on the validation set
            blobs_val = self.data_layer_val.forward()
            summary_val = self.net.get_summary(sess, blobs_val)
            self.valwriter_stage3.add_summary(summary_val, float(iter))
            last_summary_time = now
          else:
            # Compute the graph without summary
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
              self.net.train_rpn_step(sess, blobs, train_op_stage3)
          timer.toc()
          if iter == 480000:
            self.writer_stage3.close()
            self.valwriter_stage3.close()
        else:
          raise ValueError('iter is not allowed')

        # Display training information
        if iter % (cfg.TRAIN.DISPLAY) == 0:
          print('iter: %d / %d, stage: %s, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
                (iter, max_iters, stage_infor, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
          print('speed: {:.3f}s / iter'.format(timer.average_time))

        if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
          last_snapshot_iter = iter
          snapshot_path, np_path = self.snapshot(sess, iter)
          np_paths.append(np_path)
          ss_paths.append(snapshot_path)
        iter += 1

    if last_snapshot_iter != iter - 1:
      self.snapshot(sess, iter - 1)


def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def merged_networks_rfcn2rpn(origin_scope, target_scope):
  # print('####################'*10)
  # for var in tf.trainable_variables():
  #   print(var.op.name)
  # print('####################'*10)
  assign_ops = []
  for var in tf.trainable_variables():
    if (origin_scope in var.op.name):
      for var_target in tf.trainable_variables():
        if target_scope in var_target.op.name:
          if var_target.op.name.replace(target_scope, origin_scope) == var.op.name:
            assign_ops.append(tf.assign(var_target, var))
            break
  return assign_ops


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, rfcn_network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
  """Train a Fast R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    sw = SolverWrapper(sess, network, rfcn_network, imdb, roidb, valroidb, output_dir, tb_dir,
                       pretrained_model=pretrained_model)
    print('Solving...')
    sw.train_model(sess, max_iters)
    print('done solving')
