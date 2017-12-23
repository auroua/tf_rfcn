## RFCN_TensorFlow

##### Results
| model type | training strategy | mAP(%) on VOC07 test | Iterations | model_name|backbone|
|------------|:-----------------:|:--------------------:|:----------:|:---------:|:-------:|
| conv5, a trous, strides=16 without ohem| 4 stages iteration as Faster RCNN|75.77|total steps 400k satge1 80k stage2 120k stage3 80k stage4 120k|model_A|resnet_101|
| conv5, a trous, strides=16 without ohem| only training total_loss |76.35| 110k | model_B|resnet_101|

`total_loss = loss_rpn_objectness + loss_rpn_bboxes + loss_rfcn_classes + loss_rfcn_bboxes`

##### Result Details
|model_name|aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|model_A|0.8008|0.8004|0.7861|0.6579|0.4836|0.8646|0.8531|0.8774|0.6081|0.8517|0.6935|0.8884|0.8616|0.7821|0.7805|0.4693|0.7814|0.7742|0.7845|0.7516|
|model_B|0.8020|0.7940|0.7877|0.6402|0.6571|0.8599|0.8578|0.8736|0.6183|0.8223|0.6492|0.8728|0.8447|0.8201|0.7888|0.4607|0.7703|0.7558|0.8354|0.7596|

##### Training Details
**model_A**
```
  momentum: 0.9
  stage1 total steps 80k, init learning rate 0.001, step 60k learning rate 0.0001
  stage2 total steps 120k, init learning rate 0.001, step 80k learning rate 0.0001
  stage3 total steps 80k, init learning rate 0.001, step 60k learning rate 0.0001
  stage4 total steps 120k, init learning rate 0.001, step 80k learning rate 0.0001
```

**model_B**
```
  momentum: 0.9
  total steps 110k, init learning rate 0.001, step 80k learning rate 0.0001
```

##### Model Download Links
|model_name|download link|password|
|:--------:|:-----------:|:------:|
|model_A|https://pan.baidu.com/s/1jIQThtW|cgwf|
|model_B|https://pan.baidu.com/s/1i4QEVRZ|v9ua|

##### Tasks
* ~~ohem (I have tried several methods, but have no effect. The **map** in all the methods have dropped.)~~
* ~~focal loss (The focal loss also have no effect.)~~
* ~~position sensitive score map + global roi pooling class.~~
* code refactor

##### Training Pipline
1. running `tools/trainval_net_rfcn.py` file.
    * modify the net you want to use in line [import nets](https://github.com/auroua/tf_rfcn/blob/13a0892e67e474fae158ed7c0de69bf813b2ed74/tools/trainval_net_rfcn.py#L19), the nets provied by this project is in floder `lib/nets`.
    * modify the `--cfg` parameter which is the config file you want to use. Some config file can be find in file `experiments/cfgs`.
    * modify the `--weight` parameter which is the pretrained model weights file.
    * modify the `--net` which is the net architecture you want to use.
2. some other modifies:
    * you can modify the `loss function` as you requirement in line [network.py#L646](https://github.com/auroua/tf_rfcn/blob/13a0892e67e474fae158ed7c0de69bf813b2ed74/lib/nets/network.py#L646)
    * you can modify the `RFCN` architecture in line [resnet_v1_rfcn_hole.py#L279](https://github.com/auroua/tf_rfcn/blob/13a0892e67e474fae158ed7c0de69bf813b2ed74/lib/nets/resnet_v1_rfcn_hole.py#L279)_
    * you'd better using the `resnet_v1_rfcn_hole.py` and `network.py` file.


##### References:
1 [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)

2 [tensorflow-object-detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

3 [R-FCN: Object Detection via
Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409.pdf)

4 [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf)
