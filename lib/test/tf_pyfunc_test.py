import tensorflow as tf
import numpy as np


def relu(inputs):
    def _relu(x):
        return np.maximum(x, 0.)

    def _relu_grad(x):
        return np.float32(x > 0)

    def _relu_grad_op(op, grad):
        x = op.inputs[0]
        grad_x = grad*tf.py_func(_relu_grad, [x], tf.float32)
        return grad_x