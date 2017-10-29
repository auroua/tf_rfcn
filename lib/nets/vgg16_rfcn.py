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
        rpn_head = slim.conv2d(self.vgg_head, 512, [3, 3], scope='rpn_layer_head')
        rpn_anchor_scores = slim.conv2d(rpn_head, 18, [1, 1], scope='rpn_layer_scores')
        rpn_anchor_scores_shape = self._get_shape(rpn_anchor_scores)
        rpn_anchor_scores = tf.reshape(rpn_anchor_scores, shape=[rpn_anchor_scores_shape[0]*rpn_anchor_scores_shape[1]*
                                       rpn_anchor_scores_shape[2]*9, 2])
        rpn_anchor_scores_pred = tf.nn.softmax(rpn_anchor_scores)
        rpn_anchor_bboxes = slim.conv2d(self.vgg_head, 36, [1, 1], scope='rpn_layer_bboxes')
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
        rfcn_head = slim.conv2d(self.vgg_head, 512, [3, 3], scope='rfcn_layer_head')
        rfcn_cls_scores = slim.conv2d(rfcn_head, 9*21, [3, 3], scope='rfcn_layer_cls_scores')
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

        rfcn_bboxes = slim.conv2d(rfcn_head, 4*9, [1, 1], scope='rfcn_layer_bboxes')
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

    def train_op_rpn(self, scope):
        vars = self._filter_ops(scope=scope)
        self.grpn = self.optimizer_rpn.compute_gradients(self.loss_rpn, vars)

    def train_op_rfcn(self, scope):
        vars = self._filter_ops(scope=scope)
        self.grfcn = self.optimizer_rfcn.compute_gradients(self.loss_rfcn, vars)

    def train_op_rpn_stage3(self, scope):
        stage3_vars = self._get_rpn_ops(scope=scope)
        print(stage3_vars)
        self.grpn_stage3 = self.optimizer_rpn_stage3.compute_gradients(self.loss_rpn, stage3_vars)

    def train_op_rfcn_stage4(self, scope):
        stage4_vars = self._get_rfcn_ops(scope=scope)
        print(stage4_vars)
        self.grfcn_stage4 = self.optimizer_rfcn_stage4.compute_gradients(self.loss_rfcn, stage4_vars)

    def _filter_ops(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if scope not in var.op.name:
                continue
            vars.append(var)
            tf.summary.histogram(var.op.name, var)
        return vars

    def _get_rpn_ops(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if scope not in var.op.name:
                continue
            else:
                if 'rpn_layer' not in var.op.name:
                    continue
                else:
                    vars.append(var)
        return vars

    def _get_rfcn_ops(self, scope):
        vars = []
        for var in tf.trainable_variables():
            if (scope not in var.op.name) and ('rfcn_layer' not in var.op.name):
                continue
            if scope not in var.op.name:
                continue
            else:
                if 'rfcn_layer' not in var.op.name:
                    continue
                else:
                    vars.append(var)
        return vars

    def merged_networks_rpn2rfcn(self, origin_scope, target_scope):
        assign_ops = []
        for var in tf.trainable_variables():
            if (origin_scope in var.op.name) and ('rpn_layer' in var.op.name):
                for var_target in tf.trainable_variables():
                    if target_scope in var_target.op.name:
                        if var_target.op.name.replace(target_scope, origin_scope) == var.op.name:
                            assign_ops.append(tf.assign(var_target, var))
                            break
        return assign_ops

    def merged_networks_rfcn2rpn(self, origin_scope, target_scope):
        assign_ops = []
        for var in tf.trainable_variables():
            if (origin_scope in var.op.name) and ('conv' in var.op.name):
                for var_target in tf.trainable_variables():
                    if target_scope in var_target.op.name:
                        if var_target.op.name.replace(target_scope, origin_scope) == var.op.name:
                            assign_ops.append(tf.assign(var_target, var))
                            break
            if (origin_scope in var.op.name) and ('rfcn_layer' in var.op.name):
                for var_target in tf.trainable_variables():
                    if target_scope in var_target.op.name:
                        if var_target.op.name.replace(target_scope, origin_scope) == var.op.name:
                            assign_ops.append(tf.assign(var_target, var))
                            break


        return assign_ops


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
    vgg_rpn.train_op_rpn('rpn_network')
    rpn_train_op = vgg_rpn.optimizer_rpn.apply_gradients(vgg_rpn.grpn)
    vgg_rpn.train_op_rpn_stage3('rpn_network')
    vgg_rpn.train_op_rfcn_stage4('rpn_network')
    rpn_train_op_stage3 = vgg_rpn.optimizer_rpn_stage3.apply_gradients(vgg_rpn.grpn_stage3)
    rpn_train_op_stage4 = vgg_rpn.optimizer_rfcn_stage4.apply_gradients(vgg_rpn.grfcn_stage4)

    # rfcn network
    with tf.variable_scope('rfcn_network'):
        vgg_rfcn = Vgg16()
        vgg_rfcn.build_model()
        vgg_rfcn.build_rfcn(vgg_rpn.rpn_bboxes)
    vgg_rfcn.loss()
    vgg_rfcn.train_op_rfcn('rfcn_network')
    rfcn_train_op = vgg_rfcn.optimizer_rfcn.apply_gradients(vgg_rfcn.grfcn)

    for key, val in vgg_rpn.grpn:
        print(key)
        print(val)
    print('###'*30)
    for key, val in vgg_rfcn.grfcn:
        print(key)
        print(val)
    print('###'*30)
    for key, val in vgg_rpn.grpn_stage3:
        print(key)
        print(val)
    print('###'*30)
    for key, val in vgg_rpn.grfcn_stage4:
        print(key)
        print(val)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=10000)
    summary_writer = tf.summary.FileWriter(logdir=output_path, graph=sess.graph)
    merged_summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for i in range(30):
        inputs_val = np.random.normal(size=[1, 600, 1000, 3])
        intpus_val = inputs_val.astype(np.float32, copy=False)
        _, loss_rpn = sess.run([rpn_train_op, vgg_rpn.loss_rpn], feed_dict={vgg_rpn.inputs: inputs_val})
        print('step %d, loss of rpn layers %f' % (i, loss_rpn))

    print('====================================stage one finished=========================================')
    for i in range(30):
        inputs_val = np.random.normal(size=[1, 600, 1000, 3])
        intpus_val = inputs_val.astype(np.float32, copy=False)
        inputs_val_rfcn = np.random.normal(size=[1, 600, 1000, 3])
        inputs_val_rfcn = inputs_val_rfcn.astype(np.float32, copy=False)
        _, loss_rfcn, summary_ops = sess.run([rfcn_train_op, vgg_rfcn.loss_rfcn, merged_summary], feed_dict={vgg_rfcn.inputs: inputs_val_rfcn, vgg_rpn.inputs: inputs_val})
        summary_writer.add_summary(summary_ops, global_step=i)
        print('step %d, loss of rfcn layers %f' % (i, loss_rfcn))
    print('====================================stage two finished=========================================')
    # merged networks
    # merged_ops = vgg_rfcn.merged_networks_rpn2rfcn('rpn_network', 'rfcn_network')
    merged_ops = vgg_rpn.merged_networks_rfcn2rpn('rfcn_network', 'rpn_network')
    # # test merged_ops works without error
    # with tf.variable_scope('rpn_network', reuse=True):
    #     rpn_head_weights = tf.get_variable('rpn_layer_head/weights')
    #     rpn_head_biases = tf.get_variable('rpn_layer_head/biases')
    #     rpn_scores_weights = tf.get_variable('rpn_layer_scores/weights')
    #     rpn_scores_biases = tf.get_variable('rpn_layer_scores/biases')
    #     rpn_bboxes_weights = tf.get_variable('rpn_layer_bboxes/weights')
    #     rpn_bboxes_biases = tf.get_variable('rpn_layer_bboxes/biases')
    # with tf.control_dependencies(merged_ops):
    #     with tf.variable_scope('rfcn_network', reuse=True):
    #         rfcn_head_weights = tf.get_variable('rpn_layer_head/weights')
    #         rfcn_head_biases = tf.get_variable('rpn_layer_head/biases')
    #         rfcn_scores_weights = tf.get_variable('rpn_layer_scores/weights')
    #         rfcn_scores_biases = tf.get_variable('rpn_layer_scores/biases')
    #         rfcn_bboxes_weights = tf.get_variable('rpn_layer_bboxes/weights')
    #         rfcn_bboxes_biases = tf.get_variable('rpn_layer_bboxes/biases')
    #         conv1 = tf.get_variable('conv1/conv1_1/weights')
    #
    #     # rfcn_head_weights = tf.identity(rfcn_head_weights)
    #     # rfcn_head_biases = tf.identity(rfcn_head_biases)
    #     # rfcn_scores_weights = tf.identity(rfcn_scores_weights)
    #     # rfcn_scores_biases = tf.identity(rfcn_scores_biases)
    #     # rfcn_bboxes_weights = tf.identity(rfcn_bboxes_weights)
    #     # rfcn_bboxes_biases = tf.identity(rfcn_bboxes_biases)
    #
    #     conv1 = tf.identity(conv1)
    #     # with tf.control_dependencies([rfcn_head_weights, rfcn_head_biases, rfcn_scores_weights, rfcn_scores_biases,
    #     #                               rfcn_bboxes_weights, rfcn_bboxes_biases]):
    #     #     mean_weights = tf.reduce_mean(rfcn_head_weights) * 2
    #     #     mean_bias = tf.reduce_mean(rfcn_head_biases) * 2
    #
    # bool1 = tf.equal(rpn_head_biases, rfcn_head_biases)
    # bool2 = tf.equal(rpn_head_weights, rfcn_head_weights)
    # bool3 = tf.equal(rpn_scores_biases, rfcn_scores_biases)
    # bool4 = tf.equal(rpn_scores_weights, rfcn_scores_weights)
    # bool5 = tf.equal(rpn_bboxes_biases, rfcn_bboxes_biases)
    # bool6 = tf.equal(rpn_bboxes_weights, rfcn_bboxes_weights)
    #
    # # mean_weights_out = tf.reduce_mean(rpn_head_weights)
    # # mean_bias_out = tf.reduce_mean(rpn_head_biases)
    # #
    # bool_val1, bool_val2, bool_val3, bool_val4, bool_val5, bool_val6 = sess.run([bool1, bool2, bool3, bool4, bool5, bool6])
    # # # # print(bool_val1, bool_val2, bool_val3, bool_val4, bool_val5, bool_val6)
    # print(np.all(bool_val1), np.all(bool_val2), np.all(bool_val3), np.all(bool_val4), np.all(bool_val5), np.all(bool_val6))
    # weights1, bias1, weights2, bias2 = sess.run([mean_weights, mean_bias, mean_weights_out, mean_bias_out])
    # print(weights1, bias1, weights2, bias2)
    # print(weights1/weights2, bias1/bias2)
    # with tf.variable_scope('rfcn_network', reuse=True):
    #     rfcn_conv11 = tf.get_variable('conv1/conv1_1/weights')
    #     rfcn_conv12 = tf.get_variable('conv1/conv1_2/weights')
    #     rfcn_conv21 = tf.get_variable('conv2/conv2_1/weights')
    #     rfcn_conv22 = tf.get_variable('conv2/conv2_2/weights')
    #     rfcn_conv31 = tf.get_variable('conv3/conv3_1/weights')
    #     rfcn_conv32 = tf.get_variable('conv3/conv3_2/weights')
    #     rfcn_conv33 = tf.get_variable('conv3/conv3_3/weights')
    #     rfcn_conv41 = tf.get_variable('conv4/conv4_1/weights')
    #     rfcn_conv42 = tf.get_variable('conv4/conv4_2/weights')
    #     rfcn_conv43 = tf.get_variable('conv4/conv4_3/weights')
    #     rfcn_conv51 = tf.get_variable('conv5/conv5_1/weights')
    #     rfcn_conv52 = tf.get_variable('conv5/conv5_2/weights')
    #     rfcn_conv53 = tf.get_variable('conv5/conv5_3/weights')
    #     rfcn_layer_weights = tf.get_variable('rfcn_layer_head/weights')
    #     rfcn_scores_weights = tf.get_variable('rfcn_layer_cls_scores/weights')
    # rpn_conv12 = tf.get_variable('conv1/conv1_2/weights')
    # rpn_conv21 = tf.get_variable('conv2/conv2_1/weights')
    # rpn_conv22 = tf.get_variable('conv2/conv2_2/weights')
    # rpn_conv31 = tf.get_variable('conv3/conv3_1/weights')
    # rpn_conv32 = tf.get_variable('conv3/conv3_2/weights')
    # rpn_layer_weights = tf.get_variable('rfcn_layer_head/weights')
    # rpn_scores_weights = tf.get_variable('rfcn_layer_cls_scores/weights')
    # bool1 = tf.equal(rfcn_conv11, rpn_conv11)
    # bool2 = tf.equal(rfcn_conv12, rpn_conv12)
    # bool3 = tf.equal(rfcn_conv21, rpn_conv21)
    # bool4 = tf.equal(rfcn_conv22, rpn_conv22)
    # # bool5 = tf.equal(rfcn_conv31, rpn_conv31)
    # # bool6 = tf.equal(rfcn_conv32, rpn_conv32)
    # bool5 = tf.equal(rfcn_layer_weights, rpn_layer_weights)
    # bool6 = tf.equal(rfcn_scores_weights, rpn_scores_weights)
    # bool_val1, bool_val2, bool_val3, bool_val4, bool_val5, bool_val6 = sess.run(
    #     [bool1, bool2, bool3, bool4, bool5, bool6])
    # print(bool_val1, bool_val2, bool_val3, bool_val4, bool_val5, bool_val6)
    # print(np.all(bool_val1), np.all(bool_val2), np.all(bool_val3), np.all(bool_val4), np.all(bool_val5),
    #       np.all(bool_val6))
    # stage 3
    with tf.control_dependencies(merged_ops):
        with tf.variable_scope('rpn_network', reuse=True):
            rpn_conv11 = tf.get_variable('conv1/conv1_1/weights')
        rpn_conv1 = tf.identity(rpn_conv11)
        with tf.control_dependencies([rpn_conv1]):
            for i in range(100):
                inputs_val_rfcn = np.random.normal(size=[1, 600, 1000, 3])
                inputs_val_rfcn = inputs_val_rfcn.astype(np.float32, copy=False)
                _, loss_rpn_val = sess.run([rpn_train_op_stage3, vgg_rpn.loss_rpn],
                                                        feed_dict={vgg_rpn.inputs: inputs_val_rfcn})
                print('step %d, loss value is %f' %(i, loss_rpn_val))
            print('====================================stage three finished=========================================')
        for i in range(100):
            inputs_val_rfcn = np.random.normal(size=[1, 600, 1000, 3])
            inputs_val_rfcn = inputs_val_rfcn.astype(np.float32, copy=False)
            _, loss_rpn_val = sess.run([rpn_train_op_stage4, vgg_rpn.loss_rfcn],
                                                        feed_dict={vgg_rpn.inputs: inputs_val_rfcn})
            print('step %d, loss value is %f' % (i, loss_rpn_val))
        print('====================================stage four finished=========================================')
        for i in range(100):
            inputs_val_rfcn = np.random.normal(size=[1, 600, 1000, 3])
            inputs_val_rfcn = inputs_val_rfcn.astype(np.float32, copy=False)
            _, loss_rpn_val = sess.run([rpn_train_op_stage3, vgg_rpn.loss_rpn],
                                       feed_dict={vgg_rpn.inputs: inputs_val_rfcn})
            print('step %d, loss value is %f' % (i, loss_rpn_val))
        print('====================================stage five finished=========================================')
    print('finished')
    # for i in range(300):
    #     pass