import scipy.io as sio
import argparse
import collections
import pickle


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='mat file property')
  parser.add_argument('--mat_train', dest='mat_train_file',
            help='train mat file', default='/home/aurora/workspaces/data/WIDER_FACE/wider_face_split/wider_face_train.mat', type=str)
  parser.add_argument('--mat_val', dest='mat_val_file',
            help='validate mat file', default='/home/aurora/workspaces/data/WIDER_FACE/eval_tools/ground_truth/wider_face_val.mat', type=str)
  parser.add_argument('--mat_val_easy', dest='mat_val_easy',
            help='val easy mat file', default='/home/aurora/workspaces/data/WIDER_FACE/eval_tools/ground_truth/wider_easy_val.mat', type=str)
  parser.add_argument('--mat_val_medium', dest='mat_val_medium',
            help='val medium mat file', default='/home/aurora/workspaces/data/WIDER_FACE/eval_tools/ground_truth/wider_medium_val.mat', type=str)
  parser.add_argument('--mat_val_hard', dest='mat_val_hard',
            help='val hard mat', default='/home/aurora/workspaces/data/WIDER_FACE/eval_tools/ground_truth/wider_hard_val.mat', type=str)
  parser.add_argument('--mat_test', dest='mat_test',
            help='test mat', default='/home/aurora/workspaces/data/WIDER_FACE/wider_face_split/wider_face_test.mat', type=str)
  parser.add_argument('--train_image_url', dest='train_image_url',
            help='train image url', default='/home/aurora/workspaces/data/WIDER_FACE/WIDER_train/images/', type=str)
  parser.add_argument('--val_image_url', dest='val_image_url',
            help='validate image url', default='/home/aurora/workspaces/data/WIDER_FACE/WIDER_val/images/', type=str)
  parser.add_argument('--output_data_dir', dest='output_url',
            help='train output data dir', default='/home/aurora/workspaces/PycharmProjects/backup/tf-faster-rcnn/data/wider_face/', type=str)

  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit(1)

  args = parser.parse_args()
  return args


def gen_event_image_dict(mat_data, phase):
    event_list = mat_data['event_list']
    event_temp = []
    for i in range(event_list.shape[0]):
        event_temp.append(event_list[i][0][0])
    event_temp = sorted(event_temp, key=lambda x: int(x.split('--')[0]))

    files = mat_data['file_list']
    file_lists = []
    for i in range(files.shape[0]):
        # print(files[i][0].shape)
        for j in range(files[i][0].shape[0]):
            file_lists.append(files[i][0][j][0][0])
    print(file_lists)

    file_dict = collections.OrderedDict()
    for cls in event_temp:
        cls_temp = cls.split('--')
        cls_temp = cls_temp[0] + '_' + cls_temp[1]
        # print(cls_temp)
        temp_list = []
        for file_name in file_lists:
            if cls_temp in file_name:
                temp_list.append(file_name)
        file_dict[cls] = temp_list

    count = 0
    for key, value in file_dict.items():
        for val in value:
            print(key+'/'+val)
            count += 1

    with open(args.output_url+'event_image_lists_'+phase+'.pkl', 'wb') as f:
        pickle.dump(file_dict, f)


def gen_event_image_dict_without_order(mat_data, phase):
    event_list = mat_data['event_list']
    event_temp = []
    for i in range(event_list.shape[0]):
        event_temp.append(event_list[i][0][0])

    files = mat_data['file_list']
    file_lists = []
    for i in range(files.shape[0]):
        for j in range(files[i][0].shape[0]):
            file_lists.append(files[i][0][j][0][0])

    file_dict = collections.OrderedDict()
    for cls in event_temp:
        cls_temp = cls.split('--')
        cls_temp = cls_temp[0] + '_' + cls_temp[1]
        temp_list = []
        for file_name in file_lists:
            if cls_temp in file_name:
                temp_list.append(file_name)
        file_dict[cls] = temp_list

    file_event_dict = collections.OrderedDict()
    for key, value in file_dict.items():
        for val in value:
            file_event_dict[val] = key
            print(key+'/'+val)
            # count += 1

    with open(args.output_url+'image_event_dict_'+phase+'.pkl', 'wb') as f:
        pickle.dump(file_event_dict, f)


def show_dict(temp_dict):
    for key, value in temp_dict.items():
        for val in value:
            print(key+'/'+val)


def show_dict2(temp_dict):
    for key, value in temp_dict.items():
        print(key)
        print(value)


def gen_image_box_dict(mat_data, phase, level=''):
    box_list = mat_data['face_bbx_list']
    files = mat_data['file_list']
    file_image_dict = collections.OrderedDict()
    for i in range(box_list.shape[0]):
        for j in range(box_list[i][0].shape[0]):
            file_image_dict[files[i][0][j][0][0]] = box_list[i][0][j][0]
    # for key, value in file_image_dict.items():
    #     print(key)
    #     print(value)
    if level:
        with open(args.output_url+'image_box_'+phase+'_'+level+'_dict.pkl', 'wb') as f:
            pickle.dump(file_image_dict, f)
    else:
        with open(args.output_url+'image_box_'+phase+'_dict.pkl', 'wb') as f:
            pickle.dump(file_image_dict, f)


def gen_image_pose_dict(mat_data):
    pose_list = mat_data['pose_label_list']
    files = mat_data['file_list']
    file_image_dict = collections.OrderedDict()
    for i in range(pose_list.shape[0]):
        # print(box_list[i][0].shape)
        for j in range(pose_list[i][0].shape[0]):
            # print(files[i][0][j][0][0])
            # print(pose_list[i][0][j][0])
            file_image_dict[files[i][0][j][0][0]] = pose_list[i][0][j][0]
    # for key, value in file_image_dict.items():
    #     print(key)
    #     print(value)

    with open(args.train_output_data_dir+'image_pose_dict.pkl', 'wb') as f:
        pickle.dump(file_image_dict, f)


def gen_image_occlusion_dict(mat_data):
    occlusion_list = mat_data['occlusion_label_list']
    files = mat_data['file_list']
    file_image_dict = collections.OrderedDict()
    for i in range(occlusion_list.shape[0]):
        # print(box_list[i][0].shape)
        for j in range(occlusion_list[i][0].shape[0]):
            # print(files[i][0][j][0][0])
            # print(pose_list[i][0][j][0])
            file_image_dict[files[i][0][j][0][0]] = occlusion_list[i][0][j][0]
    # for key, value in file_image_dict.items():
    #     print(key)
    #     print(value)

    with open(args.train_output_data_dir+'image_occlusion_dict.pkl', 'wb') as f:
        pickle.dump(file_image_dict, f)


def gen_image_blur_dict(mat_data):
    blur_list = mat_data['blur_label_list']
    files = mat_data['file_list']
    file_image_dict = collections.OrderedDict()
    for i in range(blur_list.shape[0]):
        # print(box_list[i][0].shape)
        for j in range(blur_list[i][0].shape[0]):
            # print(files[i][0][j][0][0])
            # print(pose_list[i][0][j][0])
            file_image_dict[files[i][0][j][0][0]] = blur_list[i][0][j][0]
    # for key, value in file_image_dict.items():
    #     print(key)
    #     print(value)

    with open(args.train_output_data_dir+'image_blur_dict.pkl', 'wb') as f:
        pickle.dump(file_image_dict, f)


def gen_image_illumin_dict(mat_data):
    illumin_list = mat_data['illumination_label_list']
    files = mat_data['file_list']
    file_image_dict = collections.OrderedDict()
    for i in range(illumin_list.shape[0]):
        # print(box_list[i][0].shape)
        for j in range(illumin_list[i][0].shape[0]):
            # print(files[i][0][j][0][0])
            # print(pose_list[i][0][j][0])
            file_image_dict[files[i][0][j][0][0]] = illumin_list[i][0][j][0]
    # for key, value in file_image_dict.items():
    #     print(key)
    #     print(value)

    with open(args.train_output_data_dir+'image_illumin_dict.pkl', 'wb') as f:
        pickle.dump(file_image_dict, f)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    mat_keys = ['invalid_label_list', 'occlusion_label_list', 'pose_label_list', 'blur_label_list', 'file_list', 'event_list', 'face_bbx_list', 'illumination_label_list', 'expression_label_list']

    data_train = sio.loadmat(args.mat_train_file)
    data_val = sio.loadmat(args.mat_val_file)
    data_val_easy = sio.loadmat(args.mat_val_easy)
    data_val_medium = sio.loadmat(args.mat_val_medium)
    data_val_hard = sio.loadmat(args.mat_val_hard)
    data_test = sio.loadmat(args.mat_test)
    # with open(args.output_url+'/image_event_dict_train.pkl', 'rb') as f:
    #     event_image_dict = pickle.load(f)
    # show_dict2(event_image_dict)


    # gen_image_box_dict(data_train, 'train')
    # gen_image_box_dict(data_val, 'val')
    # gen_image_box_dict(data_val_easy, 'val', 'easy')
    # gen_image_box_dict(data_val_medium, 'val', 'medium')
    # gen_image_box_dict(data_val_hard, 'val', 'hard')

    # gen_event_image_dict_without_order(data_train, 'train')
    # gen_event_image_dict_without_order(data_val, 'val')

    # gen_event_image_dict(data_val, 'val')
    gen_event_image_dict(data_test, 'test')
