[[1708.02002] Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## Key Ideas of this paper

- The author think that one-stage object detection methods are less accurate than two-stage methods because of extrame foreground-background class imblance.
- They propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples and focus more on hard examples. 

|                          Focal-Loss                          |
| :----------------------------------------------------------: |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-30-Focal-Loss.png" width="400" />  </p> |

- A one-stage object detection method called RetinaNet is proposed based on the idea of Focal Loss. Feature Pyramid Network(FPN) is used as the backbone net and two seperate Fully Conventional Networks(FCN) are used  as subnets to predict the class and localization of objects. Focal Loss is applied to the class prediction subnet and smooth L1 loss to the localization subnet.

|                          RetinaNet                           |
| :----------------------------------------------------------: |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-30-RetinaNet.png" width="800" />  </p> |

- Results show that when trained with the Focal Loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors. 

|                   Performance of RetinaNet                   |
| :----------------------------------------------------------: |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-30-RetinaNet-Performance.png" width="400" />  </p> |



## Implementation in Pytorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)
    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    return mask.scatter_(1, index, ones)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.sum()
```



## Solutions to Class-Imbalance tasks (Comments)

- Resampling and re-design loss function are two common solutions to class-imblance tasks.
- Two-stage detection methods like Faster R-CNN applied resampling method. In RPN and Fast R-CNN part of Faster R-CNN, the foreground and background samples are kept 128:128 and 32:96.
- One stage methods like RetinaNet use all anchors to train the model, they can not select balanced samples. So try to improve the loss function seems to be the only choice.



​	



