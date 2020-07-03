import torch
import time
import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import light.model.erfnet_seg
import math
from torchvision import transforms
from PIL import Image
# torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = True

# model = get_lanenet_erfnet_seg()
model = light.model.erfnet_seg.ERFNet(1).cuda()
# model.load_state_dict(torch.load(r'erfnet_culane_best_model_exprTest-2020.05.21.15.24.24_.pth'),strict=False)
model.eval()


x = torch.zeros(( 1, 3, 256, 512),dtype=torch.float, requires_grad=False).cuda() + 1
for i in range(10):
    y = model(x)
t_all = 0
for i in range(100):
    t1 = time.time()
    y = model(x)
    t2 = time.time()
    t_all += t2 - t1

print('avg_time:',t_all / 100)
print('avg_fps:',100 / t_all)