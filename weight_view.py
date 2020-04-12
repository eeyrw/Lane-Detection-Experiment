from torchvision import transforms
from PIL import Image
import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch




original = torch.load('mobilenetv3_small_culane.pth',map_location=torch.device('cpu'))
for k,v in original.items():
    print('%s: %s %s'%(k,v.shape,v.type()))
    if v.type()=='torch.FloatTensor':
        print('     mean:%s, std:%s'%(torch.mean(v),torch.std(v)))