## RFCN_TensorFlow

##### Results
| model type | training strategy | mAP(%) on VOC07 test | Iterations | model_name|backbone|
|------------|:-----------------:|:--------------------:|:----------:|:---------:|:-------:|
| conv5, a trous, strides=16 without ohem| 4 stages iteration as Faster RCNN|75.77|40k satge1 8k stage2 12k stage3 8k stage4 12k|model_A|reanet_101|
| conv5, a trous, strides=16 without ohem| only training total_loss |76.35| 11k | model_B|resnet_101|

`total_loss = loss_rpn_objectness + loss_rpn_bboxes + loss_rfcn_classes + loss_rfcn_bboxes`

##### Details
|model_name|aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|model_A|0.8008|0.8004|0.7861|0.6579|0.4836|0.8646|0.8531|0.8774|0.6081|0.8517|0.6935|0.8884|0.8616|0.7821|0.7805|0.4693|0.7814|0.7742|0.7845|0.7516|
|model_B|0.8020|0.7940|0.7877|0.6402|0.6571|0.8599|0.8578|0.8736|0.6183|0.8223|0.6492|0.8728|0.8447|0.8201|0.7888|0.4607|0.7703|0.7558|0.8354|0.7596|

##### Model Download Links
|model_name|download link|password|
|:--------:|:-----------:|:------:|
|model_A|https://pan.baidu.com/s/1jIQThtW|cgwf|
|model_B|https://pan.baidu.com/s/1i4QEVRZ|v9ua|

##### Tasks
* OHEM

##### References:
>[tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)

> [tensorflow-object-detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

> [R-FCN: Object Detection via
Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409.pdf)
