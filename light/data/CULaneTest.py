from torch.utils.tensorboard import SummaryWriter
import numpy as np
from CULane import CULaneDataset
import collections

culaneDs = CULaneDataset('E:\CULane',
                         split='test1_crowd',
                         mode='consecutive',
                         resizeAndCropTo=(512, 256)
                         )


writer = SummaryWriter()
for a in culaneDs:
    writer.add_images('my_image_batch', a[0], 0, dataformats='NCHW')
    writer.add_images('my_seg_batch', a[1].unsqueeze(1), 0, dataformats='NCHW')
writer.close()
