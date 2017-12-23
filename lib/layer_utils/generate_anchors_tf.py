import tensorflow as tf
import numpy as np

from lib.layer_utils.generate_anchors import generate_anchors


def generate_anchors_pre_tf(height, width, feat_stride,
                            anchor_scales=(8, 16, 32),
                            anchor_ratios=(0.5, 1, 2)):
    """
    A wrapper function to generate anchors given different scales and image
    sizes in tensorflow.
    Note, since `anchor_scales` and `anchor_ratios` is in practice always
    the same, the generate 'base anchors' are static and does we only need to
    implement the shifts part in tensorflow which is depending on the image
    size.

    Parameters:
    -----------
    height: tf.Tensor
        The hight of the current image as a tensor.
    width: tf.Tensor
        The width of the current image as a tensor.
    feat_stride: tf.Tensor or scalar
        The stride used for the shifts.
    anchor_scales: list
        The scales to use for the anchors. This is a static parameter and can
        currently not be runtime dependent.
    anchor_ratios: list
        The ratios to use for the anchors. This is a static parameter and can
        currently not be runtime dependent.

    Returns:
    --------
    anchors: tf.Tensor
        2D tensor containing all anchors, it has the shape (n, 4).
    length: tf.Tensor
        A tensor containing the length 'n' of the anchors.

    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios),
                               scales=np.array(anchor_scales))

    # Calculate all shifts
    shift_x = tf.range(0, width * feat_stride, feat_stride)
    shift_y = tf.range(0, height * feat_stride, feat_stride)
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1, 1])
    shift_y = tf.reshape(shift_y, [-1, 1])
    shifts = tf.concat((shift_x, shift_y, shift_x, shift_y), 1)

    # Combine all base anchors with all shifts
    anchors = anchors[tf.newaxis] + tf.transpose(shifts[tf.newaxis], (1, 0, 2))
    anchors = tf.cast(tf.reshape(anchors, (-1, 4)), tf.float32)
    length = tf.shape(anchors)[0]

    return anchors, length
