import argparse
import pickle
import os
from mat2py import show_dict, show_dict2
import numpy as np
import cv2


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--base_data_url', dest='train_base_data',
            help='train image base files', default='/home/aurora/workspaces/data/WIDER_FACE/WIDER_train/images/', type=str)
  parser.add_argument('--file_dir', dest='image_file_dir',
            help='train output data dir', default='/home/aurora/workspaces/PycharmProjects/tensorflow/tf_rfcn/data/wider_face/image_event_dict_train.pkl', type=str)
  parser.add_argument('--box_dir', dest='box_file_dir',
            help='train output data dir', default='/home/aurora/workspaces/PycharmProjects/tensorflow/tf_rfcn/data/wider_face/image_box_train_dict.pkl', type=str)

  args = parser.parse_args()
  return args


if __name__ == '__main__':
    args = parse_args()

    with open(args.image_file_dir, 'rb') as f:
        image_event_dict = pickle.load(f)

    with open(args.box_file_dir, 'rb') as f:
        image_box_dict = pickle.load(f)

    keys_image_event = list(image_event_dict.keys())
    box_image_event = list(image_box_dict.keys())

    # for i in range(len(image_event_dict.keys())):
    #     if not keys_image_event[i]==box_image_event[i]:
    #         print(keys_image_event[i])
    images_files = ['0_Parade_marchingband_1_341',
                    '0_Parade_marchingband_1_408',
                    '0_Parade_marchingband_1_464']
    color = (0, 0, 255)
    # for i, image in enumerate(keys_image_event):
    for i, image in enumerate(images_files):
        # if i==12381:
            image_path = os.path.join(args.train_base_data, image_event_dict[image], image+'.jpg')
            # image_path = image
            print(image_path)
            img = cv2.imread(image_path)
            rois = image_box_dict[image]
            for i in range(rois.shape[0]):
                x, y, w, h = rois[i]
                # x1 = x+w
                # y1 = y+h
                # temp_x = 1024 - x
                # temp_x1 = 1024 - x1
                # if i==358:
                #     print(x, y, w, h)
                #     print('------------')
                #     print(x, y, x+w, y+h)
                # if(temp_x < temp_x1):
                #     print(i, temp_x1, y, temp_x, y1)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
            cv2.imshow('img', img)
            cv2.waitKey(0)