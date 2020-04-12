from torchvision import transforms
from PIL import Image
import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from light.utils.logger import setup_logger
from light.utils.lr_scheduler import WarmupPolyLR
from light.utils.metric import SegmentationMetric
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model
import matplotlib.pyplot as plt
import shutil
import datetime
import argparse
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



# # 定义 transforms
# transformations = transforms.Compose([transforms.ToTensor()])
# # 自定义数据集
# LaneDataset = CULaneDataset(r'E:\CULane', transform=transformations)
# # 定义 data loader
# LaneDatasetLoader = torch.utils.data.DataLoader(dataset=LaneDataset,
#                                                 batch_size=10,
#                                                 shuffle=False)


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args(bypassArgs=None):
    parser = argparse.ArgumentParser(description='Lane Detection Experiment')
    # model and dataset
    parser.add_argument('--model', type=str, default='mobilenetv3_small',
                        help='model name (default: mobilenet)')
    parser.add_argument('--dataset', type=str, default='culane',
                        help='dataset name (default: culane)')
    parser.add_argument('--rootDir', type=str, default=r'E:\CULane',
                        help='root directory (default: E:\\CULane)')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 4)')
    # cuda setting
    parser.add_argument('--cuda_usage', type=str2bool, nargs='?', default=False,
                        dest='cuda_usage', help='Whether to use CUDA.')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default='mobilenetv3_small_culane_best_model.pth',
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')

    args = parser.parse_args()

    return args


class Evaler(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        self.transformForImage = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        data_kwargs = {'transformForImage': self.transformForImage,
                       'rootDir': args.rootDir, 'requireRawImage': True}
        valset = get_segmentation_dataset(
            args.dataset, split='test8_night', **data_kwargs)
        self.val_loader = data.DataLoader(dataset=valset,
                                          shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(args.model, dataset=args.dataset,
                                            aux=args.aux, norm_layer=BatchNorm2d).to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(
                    args.resume, map_location=lambda storage, loc: storage))

    def eval(self):
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for image, target, rawImageFile in self.val_loader:
            image = image.to(self.device)
            # target = target.to(self.device)
            with torch.no_grad():
                outputs = model(image)
                aaa = torch.sigmoid(outputs[0][0][1])
                bbb = torch.sigmoid(outputs[0][0][0])

                fig = Figure(figsize=(16, 9))
                fig.tight_layout()
                canvas = FigureCanvas(fig)

                # Do some plotting.
                ax11 = fig.add_subplot(221)
                ax12 = fig.add_subplot(222)
                ax2 = fig.add_subplot(212)

                # plt.subplot(221)
                ax11.imshow(aaa, cmap=plt.cm.rainbow, vmin=0, vmax=1)
                # plt.subplot(212)
                rawImage = cv2.imread(rawImageFile[0])[..., ::-1]
                ax2.imshow(rawImage)
                ax2.imshow(aaa, alpha=aaa,cmap=plt.cm.rainbow, vmin=0, vmax=1)
                print('Current Image Path: %s' % rawImageFile)
                # plt.subplot(222)
                ax12.imshow(target[0], cmap=plt.cm.rainbow, vmin=0, vmax=1)
                # plt.show()
                canvas.draw()       # draw the canvas, cache the renderer

                buf = canvas.buffer_rgba()
                # ... convert to a NumPy array ...
                X = np.asarray(buf)[:, :, (2,1,0)]

                cv2.imshow('AAA',X)
                k = cv2.waitKey(0) & 0xff
                if k == 27:
                    break
        cv2.destroyWindow('AAA')


    def evalOnSingleImage(self, imagePath):
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        imageFile = imagePath
        rawImageRgb = Image.open(imageFile).convert('RGB')
        imageRgb = self.transformForImage(rawImageRgb)
        rawImage = torch.unsqueeze(transforms.ToTensor()(rawImageRgb), 0)
        image = torch.unsqueeze(imageRgb, 0).to(self.device)
        with torch.no_grad():
            outputs = model(image)
            aaa = torch.sigmoid(outputs[0][0][1])
            bbb = torch.sigmoid(outputs[0][0][0])
            plt.subplot(211)
            plt.imshow(aaa, cmap=plt.cm.hot, vmin=torch.min(
                aaa), vmax=torch.max(aaa))
            # plt.subplot(222)
            # plt.imshow(target[0].numpy(), cmap=plt.cm.hot, vmin=0, vmax=1)
            # plt.subplot(223)
            # plt.imshow(image[0].permute(1,2,0))
            plt.subplot(212)
            plt.imshow(rawImage[0].permute(1, 2, 0))
            plt.show()


if __name__ == '__main__':

    print(os.getcwd())
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if args.cuda_usage and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"

    evaler = Evaler(args)
    # evaler.eval()
    evaler.evalOnSingleImage(r"D:\tusimple\train_set\clips\0601\1494452439568910656\2.jpg")
    torch.cuda.empty_cache()
