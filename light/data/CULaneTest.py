from torch.utils.tensorboard import SummaryWriter
import numpy as np
from CULane import CULaneDataset
import collections
import os
import sys
import sync_transforms as pairedTr
import torch

# image transform
transfrom_all = pairedTr.Compose([
    pairedTr.RandomPerspective(distortion_scale=0.3,p=0.2),
    pairedTr.RandomResizedCrop((256, 512), scale=(0.6, 1.0), ratio=(2/1, 2/1)),
])

input_transform = pairedTr.Compose([
    pairedTr.ColorJitter(0.3, 0.3, 0.3),
    pairedTr.ToTensor(),
    pairedTr.RandomErasing(p=0.1),
    pairedTr.Normalize([.485, .456, .406], [.229, .224, .225]),
])

transfrom_seg = None

culaneDs = CULaneDataset('E:\CULane',
                         split='test',
                         mode='consecutive',
                         #resizeAndCropTo=(512, 256),
                         framesGroupSize=10,
                         transformForImage=input_transform,
                         transformForSeg=transfrom_seg,
                         transformForAll=transfrom_all,
                         segDistinguishInstance=True
                         )


writer = SummaryWriter()


for i in range(453, 453+20):
    a = culaneDs[i]
    maxVal=torch.max(a[0])
    minVal=torch.min(a[0])
    b=(a[0]-minVal)/(maxVal-minVal)
    maxVal=torch.max(a[1])
    minVal=torch.min(a[1])
    d=(a[1].float()-minVal)/(maxVal-minVal)
    writer.add_images('my_image_batch', b, 0, dataformats='NCHW')
    writer.add_images('my_seg_batch', d, 0, dataformats='NCHW')
writer.close()
