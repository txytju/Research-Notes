[[1312.6229] OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/abs/1312.6229)

## Meta Information

- author : LeCun group
- Year : 2013

## Key Ideas

1. Shared features for multi-task learning. AlexNet was improved by this paper for multi-task learning. Features learned with classification task can be easilly applied to localization and detection tasks by justing changing the last few layers of the network.

2. Idea of fully convolutional network.

   ![](http://p3rz3gu1u.bkt.clouddn.com/2018-07-02-025309.png)

3. Sliding window on feature map rather than raw image, which saved a lot of computation compared with raw-image-sliding-window methods. The idea of using feature map information are used a lot in latter methods like Fast R-CNN.

## Shortcomings

1. Multi-scale sliding window method was applied, which led to too much computation.
2. The backbone network (OverFeat) was sitll too weak with not enough ability for feature representation.

## Historical position

- This is an old paper (2014) and has been replaced by SSD and YOLO for more realtime detections. 
- However the paper does provide interesting perspective. 
- On the ILSVRC 2013 dataset OverFeat ranked 4th in classification, 1st in localization, and 1st in detection.

