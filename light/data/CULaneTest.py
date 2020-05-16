from torch.utils.tensorboard import SummaryWriter
import numpy as np
from CULane import CULaneDataset
import collections
import os
import sys
import sync_transforms as pairedTr

# image transform
transfrom_all = pairedTr.Compose([
    pairedTr.RandomPerspective(distortion_scale=0.3,p=0.2),
    pairedTr.RandomResizedCrop((256, 512), scale=(0.6, 1.0), ratio=(2/1, 2/1)),
])

input_transform = pairedTr.Compose([
    pairedTr.ColorJitter(0.3, 0.3, 0.3),
    pairedTr.ToTensor(),
    pairedTr.RandomErasing(p=0.1),
    # pairedTr.Normalize([.485, .456, .406], [.229, .224, .225]),
])

transfrom_seg = pairedTr.Compose([
    pairedTr.ToTensor(),
    # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

culaneDs = CULaneDataset('E:\CULane',
                         split='test',
                         mode='consecutive',
                         #resizeAndCropTo=(512, 256),
                         framesGroupSize=5,
                         transformForImage=input_transform,
                         transformForSeg=transfrom_seg,
                         transformForAll=transfrom_all,
                         segDistinguishInstance=True
                         )


writer = SummaryWriter()


for i in range(0, 2):
    a = culaneDs[i]
    writer.add_images('my_image_batch', a[0], 0, dataformats='NCHW')
    writer.add_images('my_seg_batch', a[1].unsqueeze(1), 0, dataformats='NCHW')
writer.close()
