import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow


if __name__ == '__main__':
    # ckpt_path = '/home/aurora/workspaces/PycharmProjects/tensorflow/tf_rfcn/output/res101/voc_2007_trainval+voc_2012_' \
    #             'trainval/default/res101_faster_rcnn_iter_10.ckpt'
    # ckpt_path = '/home/aurora/workspaces/PycharmProjects/tensorflow/tf_rfcn/output/res101/voc_2007_trainval+voc_2012_' \
    #             'trainval/default/res101_faster_rcnn_iter_400000.ckpt'
    ckpt_path = '/home/aurora/pretrained_models/tensorflow_backbone_network/resnet_v1_50.ckpt'
    ckpt_path = '/home/aurora/workspaces/PycharmProjects/tensorflow/tf_rfcn/output/res101_global_local' \
                '/voc_2007_trainval+voc_2012_trainval/local_global_7721/res101_rfcn_local_global_iter_110000.ckpt'
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        count = 0
        for key in var_to_shape_map:
            # if 'rfcn_network/resnet_v1_101' in key and 'rfcn_network/resnet_v1_101/block' not in key:
            # if 'rpn_network/resnet_v1_101/block2/unit_1' in key:
            # if 'resnet_v1_101/conv1' in key:
                print(key)
                count += 1
        print(count)
    except Exception as e:
        print(str(e))