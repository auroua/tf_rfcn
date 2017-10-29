import tensorflow as tf
import argparse
import numpy as np
import tensorflow.contrib.slim as slim


def init_parameter():
    parser = argparse.ArgumentParser(description='VGG16')
    parser.add_argument('--batch_size', action='store_true', help='batch size')

    args = parser.parse_args()
    return args


class Vgg16():
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, shape=[1, 600, 1000, 3], name='images')
        self.K = 3
        self.optimizer_rpn = tf.train.MomentumOptimizer(0.01, 0.9)
        self.optimizer_rfcn = tf.train.MomentumOptimizer(0.01, 0.9)
        self.optimizer_rpn_stage3 = tf.train.MomentumOptimizer(0.01, 0.9)
        self.optimizer_rfcn_stage4 = tf.train.MomentumOptimizer(0.01, 0.9)

    def build_backbones(self):
        inputs = self.inputs
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            padding='SAME', weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=tf.nn.relu):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            self.vgg_head = net

    def build_rpn(self):
        rpn_head = slim.conv2d(self.vgg_head, 512, [3, 3], scope='rpn_head')
        rpn_anchor_scores = slim.conv2d(rpn_head, 18, [1, 1], scope='rpn_scores')
        rpn_anchor_scores_shape = self._get_shape(rpn_anchor_scores)
        rpn_anchor_scores = tf.reshape(rpn_anchor_scores, shape=[rpn_anchor_scores_shape[0]*rpn_anchor_scores_shape[1]*
                                       rpn_anchor_scores_shape[2]*9, 2])
        rpn_anchor_scores_pred = tf.nn.softmax(rpn_anchor_scores)
        rpn_anchor_bboxes = slim.conv2d(self.vgg_head, 36, [1, 1], scope='rpn_bboxes')
        rpn_anchor_bboxes_shape = self._get_shape(rpn_anchor_bboxes)
        rpn_anchor_bboxes = tf.reshape(rpn_anchor_bboxes, shape=[rpn_anchor_bboxes_shape[0]*rpn_anchor_bboxes_shape[1]*
                                                                 rpn_anchor_bboxes_shape[2]*9, -1])
        self.rpn_scores = rpn_anchor_scores
        self.rpn_bboxes = rpn_anchor_bboxes
        self.rpn_pred = rpn_anchor_scores_pred

    def build_rfcn(self, rois):
        k = self.K
        step_x = (rois[:, 2] - rois[:, 0])/k
        step_y = (rois[:, 3] - rois[:, 1])/k

        inputs_height = self.inputs.shape.as_list()[1]
        inputs_width = self.inputs.shape.as_list()[2]

        total_boxes = []
        for y in range(k):
            for x in range(k):
                ymin = (rois[:, 1] + y*step_y)/inputs_height
                xmin = (rois[:, 0] + x*step_x)/inputs_width
                ymax = (rois[:, 1] + (y+1)*step_y)/inputs_height
                xmax = (rois[:, 0] + (x+1)*step_x)/inputs_width
                boxes = tf.stop_gradient(tf.stack([ymin, xmin, ymax, xmax], axis=1))
                total_boxes.append(boxes)

        # rfcn conv_layer
        rfcn_head = slim.conv2d(self.vgg_head, 512, [3, 3], scope='rfcn_head')
        rfcn_cls_scores = slim.conv2d(rfcn_head, 9*21, [3, 3], scope='rfcn_cls_scores')
        rfcn_scores_collections = tf.split(rfcn_cls_scores, 9, axis=3)

        rpn_crops = []
        for boxes, scores_feature in zip(total_boxes, rfcn_scores_collections):
            crops = tf.image.crop_and_resize(scores_feature, boxes,
                                             box_ind=[0] * 5022,
                                             crop_size=[3, 3])
            rpn_crops.append(crops)
        avg_crops = tf.add_n(rpn_crops)/len(rpn_crops)
        rpn_scores = tf.reduce_mean(avg_crops, axis=[1, 2])
        rpn_pred = tf.nn.softmax(rpn_scores)

        rfcn_bboxes = slim.conv2d(rfcn_head, 4*9, [1, 1], scope='rfcn_bboxes')
        rfcn_bboxes_regions = tf.split(rfcn_bboxes, 9, axis=3)
        total_boxes_crops = []
        for bboxes, features in zip(total_boxes, rfcn_bboxes_regions):
            crops = tf.image.crop_and_resize(features, bboxes,
                                             box_ind=[0] * 5022,
                                             crop_size=[3, 3])
            total_boxes_crops.append(crops)
        avg_bboxes = tf.add_n(total_boxes_crops)/len(total_boxes_crops)
        avg_bboxes_scores = tf.reduce_mean(avg_bboxes, axis=[1, 2])

        self.rfcn_pred = rpn_pred
        self.rfcn_bbox_scores = avg_bboxes_scores

    def build_model(self):
        self.build_backbones()
        self.build_rpn()

    def _get_shape(self, layers_input):
        static_shape = layers_input.shape.as_list()
        dynamic = tf.shape(layers_input)
        shapes = [dims[0] if dims[0] is not None else dims[1] for dims in zip(static_shape, tf.unstack(dynamic))]
        return shapes

    def loss(self):
        loss_rpn = tf.nn.l2_loss(tf.reduce_mean(self.rpn_pred) + tf.reduce_mean(self.rpn_bboxes) - 5)
        loss_rfcn = tf.nn.l2_loss(tf.reduce_mean(self.rfcn_bbox_scores) + tf.reduce_mean(self.rfcn_pred) - 3)
        tf.summary.scalar('loss_rpn', loss_rpn)
        tf.summary.scalar('loss_rfcn', loss_rfcn)
        self.loss_rpn = loss_rpn
        self.loss_rfcn = loss_rfcn

    def train_op(self, scope):
        vars = self._filter_ops(scope=scope)
        stage3_vars = self._get_rpn_ops(scope=scope)
        print(stage3_vars)
        stage4_vars = self._get_rfcn_ops(scope=scope)
        self.grpn = self.optimizer_rpn.compute_gradients(self.loss_rpn, vars)
        self.grfcn = self.optimizer_rfcn.compute_gradients(self.loss_rfcn, vars)
        self.grpn_stage3 = self.optimizer_rpn_stage3.compute_gradients(self.loss_rpn, stage3_vars)
        self.grfcn_stage4 = self.optimizer_rfcn_stage4.compute_gradients(self.loss_rfcn, stage4_vars)

    def _filter_ops(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if scope not in var.op.name:
                continue
            vars.append(var)
            # tf.summary.histogram(var.op.name, var)
        return vars

    def _get_rpn_ops(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if (scope not in var.op.name) and ('rpn' not in var.op.name):
                continue
            vars.append(var)
        return vars

    def _get_rfcn_ops(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if (scope not in var.op.name) and ('rfcn' not in var.op.name):
                continue
            vars.append(var)
        return vars


if __name__ == '__main__':
    output_path = '/home/aurora/workspaces/PycharmProjects/tensorflow/tf_rfcn/output/vgg_test'

    # rpn network
    with tf.variable_scope('rpn_network'):
        vgg_rpn = Vgg16()
        vgg_rpn.build_model()
        vgg_rpn.build_rfcn(vgg_rpn.rpn_bboxes)
    print(vgg_rpn.vgg_head.shape.as_list())
    print(vgg_rpn.rpn_bboxes.shape.as_list())
    print(vgg_rpn.rpn_pred.shape.as_list())
    print(vgg_rpn.rfcn_pred.shape.as_list())
    print(vgg_rpn.rfcn_bbox_scores.shape.as_list())
    print('==='*30)
    vgg_rpn.loss()
    vgg_rpn.train_op('rpn_network')
    rpn_train_op = vgg_rpn.optimizer_rpn.apply_gradients(vgg_rpn.grpn)

    # rfcn network
    with tf.variable_scope('rfcn_network'):
        vgg_rfcn = Vgg16()
        vgg_rfcn.build_model()
        vgg_rfcn.build_rfcn(vgg_rpn.rpn_bboxes)
    vgg_rfcn.loss()
    vgg_rfcn.train_op('rfcn_network')
    rfcn_train_op = vgg_rfcn.optimizer_rfcn.apply_gradients(vgg_rfcn.grfcn)

    # for key, val in vgg_rpn.grpn:
    #     print(key)
    #     print(val)
    # print('###'*30)

    for key, val in vgg_rpn.grpn_stage3:
        print(key)
        print(val)
    print('###' * 30)
    # for key, val in vgg_rpn.grfcn_stage4:
    #     print(key)
    #     print(val)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=10000)
    summary_writer = tf.summary.FileWriter(logdir=output_path, graph=sess.graph)
    merged_summary = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    with tf.variable_scope('rpn_network', reuse=True):
        var = tf.get_variable('rpn_bboxes/weights')

    for i in range(300):
        inputs_val = np.random.normal(size=[1, 600, 1000, 3])
        intpus_val = inputs_val.astype(np.float32, copy=False)
        _, loss_rpn = sess.run([rpn_train_op, vgg_rpn.loss_rpn], feed_dict={vgg_rpn.inputs: inputs_val})
        # _, loss_rpn, summary_ops = sess.run([rpn_train_op, vgg_rpn.loss_rpn, merged_summary], feed_dict={vgg_rpn.inputs: inputs_val})
        # summary_writer.add_summary(summary_ops, global_step=i)
        print('step %d, loss of rpn layers %f' % (i, loss_rpn))

    print('====================================stage one finished=========================================')
    for i in range(300):
        inputs_val = np.random.normal(size=[1, 600, 1000, 3])
        intpus_val = inputs_val.astype(np.float32, copy=False)
        inputs_val_rfcn = np.random.normal(size=[1, 600, 1000, 3])
        inputs_val_rfcn = inputs_val_rfcn.astype(np.float32, copy=False)
        _, loss_rfcn, summary_ops = sess.run([rfcn_train_op, vgg_rfcn.loss_rfcn, merged_summary], feed_dict={vgg_rfcn.inputs: inputs_val_rfcn, vgg_rpn.inputs: inputs_val})
        summary_writer.add_summary(summary_ops, global_step=i)
        print('step %d, loss of rfcn layers %f' % (i, loss_rfcn))
    # print('====================================stage two finished=========================================')
    # for i in range(300):
    #     pass



