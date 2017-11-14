import tensorflow as tf
from tensorflow.python.ops import array_ops

slim = tf.contrib.slim

def focal_loss(onehot_labels, cls_preds,
                            alpha=0.25, gamma=2.0, name=None, scope=None):
    """Compute softmax focal loss between logits and onehot labels
    logits and onehot_labels must have same shape [batchsize, num_classes] and
    the same data type (float16, 32, 64)
    Args:
      onehot_labels: Each row labels[i] must be a valid probability distribution
      cls_preds: Unscaled log probabilities
      alpha: The hyperparameter for adjusting biased samples, default is 0.25
      gamma: The hyperparameter for penalizing the easy labeled samples
      name: A name for the operation (optional)
    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
        logits = tf.convert_to_tensor(cls_preds)
        onehot_labels = tf.convert_to_tensor(onehot_labels)

        precise_logits = tf.cast(logits, tf.float32) if (
                        logits.dtype == tf.float16) else logits
        onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
        predictions = tf.nn.sigmoid(logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
        # add small value to avoid 0
        epsilon = 1e-8
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * onehot_labels * tf.log(predictions_pt+epsilon),
                                     name=name, axis=1)
        return losses


def focal_loss2(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent, axis=1)


def focal_loss3(cls_score, label, num_classes):
    alpha_scale = 0.25
    gamma = 2
    epsilon = 1e-8
    label = tf.one_hot(label, depth=num_classes)
    cls_pred = tf.nn.sigmoid(cls_score)
    predictions_pt = tf.where(tf.equal(label, 1), cls_pred, 1 - cls_pred)
    alpha_t = tf.ones_like(label, dtype=tf.float32)
    alpha_t = tf.scalar_mul(alpha_scale, alpha_t)
    alpha_t = tf.where(tf.equal(label, 1.0), alpha_t, 1. - alpha_t)
    losses = tf.reduce_mean(-alpha_t * tf.pow(1 - predictions_pt, gamma) * tf.log(predictions_pt + epsilon), axis=1)
    return losses


def regression_loss(pred_boxes, gt_boxes, weights):
    """
    Regression loss (Smooth L1 loss: also known as huber loss)
    Args:
        pred_boxes: [# anchors, 4]
        gt_boxes: [# anchors, 4]
        weights: Tensor of weights multiplied by loss with shape [# anchors]
    """
    loss = tf.losses.huber_loss(predictions=pred_boxes, labels=gt_boxes,
                                weights=weights, scope='box_loss')
    return loss


def test():
    logits = tf.convert_to_tensor([[0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2]])
    logits_sigmoid = tf.nn.sigmoid(logits)
    labels = slim.one_hot_encoding([1, 2], 4)
    labels_vector = tf.constant([1, 2])
    bbox = tf.ones_like(logits)
    with tf.Session() as sess:
        print(sess.run(logits))
        print(sess.run(logits_sigmoid))
        print(sess.run(focal_loss(onehot_labels=labels, cls_preds=logits)))
        print(sess.run(focal_loss2(target_tensor=labels, prediction_tensor=logits)))
        print(sess.run(focal_loss3(cls_score=logits, label=labels_vector, num_classes=4)))
        print(sess.run(regression_loss(logits, bbox, tf.expand_dims(1./tf.convert_to_tensor([2, 3], dtype=tf.float32), 1))))
    sess.close()

test()