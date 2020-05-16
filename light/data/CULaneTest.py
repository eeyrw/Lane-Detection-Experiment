from torch.utils.tensorboard import SummaryWriter
import numpy as np
from CULane import CULaneDataset
import collections

culaneDs = CULaneDataset('E:\CULane',
                         split='test',
                         mode='consecutive',
                         resizeAndCropTo=(512, 256),
                         framesGroupSize=5
                         )


writer = SummaryWriter()

a=culaneDs[32]
writer.add_images('my_image_batch', a[0], 0, dataformats='NCHW')
writer.add_images('my_seg_batch', a[1].unsqueeze(1), 0, dataformats='NCHW')
writer.close()
