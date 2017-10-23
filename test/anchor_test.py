import tensorflow as tf
import numpy as np
from layer_utils import snippets


if __name__ == '__main__':
    anchors, length = snippets.generate_anchors_pre(40, 60, 16)
    print(anchors[:32, :])
    print(length)
