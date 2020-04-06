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
from light.utils.distributed import *
from light.utils.logger import setup_logger
from light.utils.lr_scheduler import WarmupPolyLR
from light.utils.metric import SegmentationMetric
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model
from light.nn import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss, SoftDiceLoss
import matplotlib.pyplot as plt
import shutil
import datetime
import argparse
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time

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
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=240, metavar='N',
                        help='number of epochs to train (default: 240)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--cuda_usage', type=str2bool, nargs='?', default=False,
                        dest='cuda_usage', help='Whether to use CUDA.')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default='./mobilenetv3_small_culane.pth',
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    if bypassArgs:
        args = parser.parse_args(bypassArgs)
    else:
        args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    args.lr = args.lr / 4 * args.batch_size

    return args


class Evaler(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        data_kwargs = {'transform': input_transform, 'rootDir': args.rootDir}
        if not args.skip_val:
            valset = get_segmentation_dataset(
                args.dataset, split='train', mode='val', **data_kwargs)
            val_sampler = make_data_sampler(valset, False, args.distributed)
            val_batch_sampler = make_batch_data_sampler(
                val_sampler, args.batch_size)
            self.val_loader = data.DataLoader(dataset=valset,
                                              batch_sampler=val_batch_sampler,
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

        # create criterion
        self.criterion = MixSoftmaxCrossEntropyLoss(
            args.aux, args.aux_weight, ignore_index=-1).to(self.device)
        # self.criterion = SoftDiceLoss().to(self.device)


        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[
                                                                 args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric = SegmentationMetric(valset.num_class)

        self.best_pred = 0.0

    def eval(self):
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target,rawSeg) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
                aaa=outputs[0][0][1]
                bbb=outputs[0][0][0]
                plt.subplot(311)
                plt.imshow(aaa, cmap=plt.cm.hot, vmin=torch.min(aaa), vmax=torch.max(aaa))
                plt.subplot(312)
                plt.imshow(bbb, cmap=plt.cm.hot, vmin=torch.min(bbb), vmax=torch.max(bbb))
                plt.subplot(313)
                plt.imshow(target[0].numpy(),cmap=plt.cm.hot, vmin=0, vmax=2)
                plt.colorbar()
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
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    logger = setup_logger(args.model, args.log_dir, get_rank(), filename='{}_{}_log.txt'.format(
        args.model, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    evaler = Evaler(args)
    evaler.eval()
    torch.cuda.empty_cache()
