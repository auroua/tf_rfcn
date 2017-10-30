import tensorflow as tf
import numpy as np
import cv2
from nets.resnet_v1_rfcn_hole import resnetv1
import argparse
import sys
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import pprint
import os
from utils.blob import im_list_to_blob
from model.bbox_transform import clip_boxes, bbox_transform_inv
import time
from model.test import im_detect
from model.config import cfg, get_output_dir
from utils.cython_nms import nms, nms_new
from model.test import test_net

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
            help='model to test',
            default=None, type=str)
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes



def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors



def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    # if has model, get the name from it
    # if does not, then just use the initialization weights
    if args.model:
        filename = os.path.splitext(os.path.basename(args.model))[0]
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]

    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)



    net = resnetv1(num_layers=101)

    # load model
    net.create_architecture(sess, "TEST", imdb.num_classes, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

    print(('Loading model check point from {:s}').format(args.model))
    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)
    # for var in tf.all_variables():
    #     print(var.op.name)
    #     print(sess.run(var))
    # np.random.seed(cfg.RNG_SEED)
    # """Test a Fast R-CNN network on an image database."""
    # num_images = len(imdb.image_index)
    # # all detections are collected into:
    # #  all_boxes[cls][image] = N x 5 array of detections in
    # #  (x1, y1, x2, y2, score)
    # all_boxes = [[[] for _ in range(num_images)]
    #              for _ in range(imdb.num_classes)]
    #
    # output_dir = get_output_dir(imdb, filename)
    # # im = cv2.imread('/home/aurora/workspaces/data/voc/voc2007/VOC2007/JPEGImages/006258.jpg')
    # # im = cv2.imread('/home/aurora/workspaces/PycharmProjects/tensorflow/tf_rfcn/data/VOCdevkit2007/VOC2007/JPEGImages/000001.jpg')
    # _t = {'im_detect': Timer(), 'misc': Timer()}
    #
    # _t['im_detect'].tic()
    # scores, boxes = im_detect(sess, net, im)
    # _t['im_detect'].toc()
    # print(scores)
    # print(boxes)
    # thresh = 0.05
    # max_per_image = 100
    # for i in range(num_images):
    #     im = cv2.imread(imdb.image_path_at(i))
    #     # print('==='*10)
    #     # print(imdb.image_path_at(i))
    #
    #     # cv2.imshow('input_img', im)
    #     # cv2.waitKey(0)
    #
    #     _t['im_detect'].tic()
    #     scores, boxes = im_detect(sess, net, im)
    #     _t['im_detect'].toc()
    #
    #     # print(scores)
    #     # print(boxes)
    #     _t['misc'].tic()
    #
    #     # skip j = 0, because it's the background class
    #     for j in range(1, imdb.num_classes):
    #         inds = np.where(scores[:, j] > thresh)[0]
    #         cls_scores = scores[inds, j]
    #         cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
    #         cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
    #             .astype(np.float32, copy=False)
    #         keep = nms(cls_dets, cfg.TEST.NMS)
    #         cls_dets = cls_dets[keep, :]
    #         all_boxes[j][i] = cls_dets
    #
    #     # Limit to max_per_image detections *over all classes*
    #     if max_per_image > 0:
    #         image_scores = np.hstack([all_boxes[j][i][:, -1]
    #                                   for j in range(1, imdb.num_classes)])
    #         if len(image_scores) > max_per_image:
    #             image_thresh = np.sort(image_scores)[-max_per_image]
    #             for j in range(1, imdb.num_classes):
    #                 keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
    #                 all_boxes[j][i] = all_boxes[j][i][keep, :]
    #     _t['misc'].toc()
    #
    #     print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
    #           .format(i + 1, num_images, _t['im_detect'].average_time,
    #                   _t['misc'].average_time))