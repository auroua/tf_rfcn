import numpy as np
import tensorflow as tf


def _normalize(rois, height, width):
    xmin, ymin, xmax, ymax = tf.unstack(rois, axis=1)
    xmin = xmin/width
    ymin = ymin/height
    xmax = xmax/width
    ymax = ymax/height
    return tf.stack([xmin, ymin, xmax, ymax], axis=1)


def _grid_features(imgs, tf_rois):
    bins = 3
    xmin, ymin, xmax, ymax = tf.unstack(tf_rois, axis=1)
    # xmin = tf.slice(tf_rois, [0, 0], [-1, 1], name='xmin')
    # ymin = tf.slice(tf_rois, [0, 1], [-1, 1], name='ymin')
    # xmax = tf.slice(tf_rois, [0, 2], [-1, 1], name='xmax')
    # ymax = tf.slice(tf_rois, [0, 3], [-1, 1], name='ymax')
    print(xmin)
    stepx = (xmax - xmin) / bins
    stepy = (ymax - ymin) / bins
    gt_rois = []
    for biny in range(bins):
        for binx in range(bins):
            sub_rois = [ymin + stepy*biny,
                        xmin + stepx*binx,
                        ymin + stepy*(biny + 1),
                        xmin + stepx*(binx + 1)]
            tf_sub_rois = tf.stack(sub_rois, axis=1)
            gt_rois.append(tf_sub_rois)
    total_crops = []
    for rois in gt_rois:
        img_crop = tf.image.crop_and_resize(imgs, rois, box_ind=tf.to_int32([0, 0]), crop_size=[2, 2])
        total_crops.append(img_crop)

    return total_crops, gt_rois


if __name__ == '__main__':
    np.random.seed(0)

    img = np.random.randn(1, 10, 10, 1)
    tf_img = tf.convert_to_tensor(img)
    print(img[0, :, :, 0])
    rois = np.array([[0, 0, 6, 6],
                     [2, 2, 8, 8]], dtype=np.float32)
    tf_rois = tf.convert_to_tensor(rois, dtype=tf.float32)
    tf_rois_norm = _normalize(rois, 10, 10)
    total_crops_tf, gt_rois = _grid_features(tf_img, tf_rois_norm)

    sess = tf.Session()

    # print(sess.run(tf_rois_norm))
    # for gt_roi in gt_rois:
    #     print(sess.run(gt_roi))

    for crop in total_crops_tf:
        print(sess.run(tf.squeeze(crop, axis=[3])))
        print('==='*10)

