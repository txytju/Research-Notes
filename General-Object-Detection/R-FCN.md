[[1605.06409] R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)

## Meta Information

- author : MSRA
- Conference : NIPS
- Year : 2016

## Key Ideas

- Two-stage object detection methods like Faster R-CNN split the pipeline into 2 parts : 1) a shared fully convolutional subnetwork independent of ROIs; 2) an ROI-wise subnetwork that do not share computation. This kind of design has 2 shortcomes :
  - The computation after ROIs are not shared, which lead to much computation.
  - A dilemma of increasing translation invariance for image classification vs. respecting translation variance for object detection. 
- In this paper, two new parts are introduced :
  - They used extra conv layer called **position-sensitive score maps** to include more locolization information. Each of these score maps encodes the position information with respect to a **relative spatial position**. 
  - A **position-sensitive RoI pooling layer** is used instead of ordinary ROI pooling.
- This paper achieve better accuracy and less computation time than original Faster R-CNN.

![](http://p3rz3gu1u.bkt.clouddn.com/2018-07-02-R-FCN.png)

