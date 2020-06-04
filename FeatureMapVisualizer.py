import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import light.model.erfnet_seg
import math


class FeatureMapVisualizer(object):
    def __init__(self, model):
        self.model = model
        self._HookModel()
        self.counter = 0

    def viz(self, module, input):
        x = input[0][0]
        # 最多显示4张图
        min_num = np.minimum(128, x.size()[0])
        col = int(math.sqrt(min_num))
        col = col if min_num-col*col <= 0 else col+1
        row = col if min_num-col*col <= 0 else col+1
        for i in range(min_num):
            plt.subplot(row, col, i+1)
            plt.imshow(x[i].cpu().detach().numpy())
        # plt.show()
        plt.savefig('FeatureMaps/%s_%d.png' % ('sss', self.counter))
        self.counter += 1

    def _HookModel(self):
        for name, m in self.model.named_modules():
            # if not isinstance(m, torch.nn.ModuleList) and \
            #         not isinstance(m, torch.nn.Sequential) and \
            #         type(m) in torch.nn.__dict__.values():
            # 这里只对卷积层的feature map进行显示
            if isinstance(m, torch.nn.Conv2d):
                m.register_forward_pre_hook(self.viz)


if __name__ == '__main__':
    # model = get_lanenet_erfnet_seg()
    model = light.model.erfnet_seg.ERFNet(1)
    vs = FeatureMapVisualizer(model)
    batchInputs = torch.randn(
        1, 3, 128, 256, dtype=torch.float, requires_grad=False)
    b = model(batchInputs)
