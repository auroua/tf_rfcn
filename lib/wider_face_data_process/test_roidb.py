import pickle

if __name__ == '__main__':
    with open('/home/aurora/workspaces/PycharmProjects/backup/tf-faster-rcnn/data/cache/wider_face_train_gt_roidb.pkl','rb') as f:
        datas = pickle.load(f)

    print(datas[0])
    for data in datas:
        if data['test_image_names'] == '41_Swimming_Swimmer_41_279':
            print(data['boxes'])