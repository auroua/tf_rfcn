import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 2
    iou_threshold_np = 0.5

    boxes = tf.constant(boxes_np)
    scores = tf.constant(scores_np)
    max_output_size = tf.constant(max_output_size_np)
    # iou_threshold = tf.constant(iou_threshold_np, dtype=tf.float32)
    keep_index = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold_np)

    sess = tf.Session()
    print(sess.run(keep_index))