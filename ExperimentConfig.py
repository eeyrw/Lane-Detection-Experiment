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

import shutil
import datetime
import argparse
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time
from parameter import getParameter
from torch.utils.tensorboard import SummaryWriter
import light.data.sync_transforms as pairedTr
import ExperimentHelper


class ExperimentConfig(object):
    def __init__(self):
        # experiment name (default: defaultExpr)
        self.exprName = 'defaultExpr'

        # model name (default: erfnet)
        self.modelName = 'erfnet'
        # pretrain weight file path (default: None )
        self.pretrainWeight = None

        # dataset name (default: culane)
        self.dataset = 'culane'
        # root directory (default: E:\\CULane)
        self.rootDir = r'E:\CULane'
        # crop image size height
        self.crop_size_h = 256
        # crop image size width
        self.crop_size_w = 512
        # dataloader threads
        self.workers = 4
        # input batch size for training (default: 4)
        self.batch_size = 2

        # start epochs (default:0)
        self.start_epoch = 0
        # number of epochs to train (default: 240)
        self.epochs = 240
        # learning rate (default: 1e-4)
        self.lr = 1e-4
        # momentum (default: 0.9)
        self.momentum = 0.9
        # w-decay (default: 5e-4)
        self.weight_decay = 5e-4
        # warmup iters
        self.warmup_iters = 0
        # lr = warmup_factor * lr
        self.warmup_factor = 1.0 / 3
        # method of warmup
        self.warmup_method = 'linear'
        # multistep lr step size
        self.step_size = 10000
        # Whether to use CUDA.
        self.cuda_usage = True
        #
        self.local_rank = 0
        # put the path to resuming file if needed
        self.resume = None
        # Directory for saving checkpoint models
        self.save_dir = '~/.torch/models'
        # save model every checkpoint-epoch
        self.save_epoch = 10
        # Directory for saving checkpoint models
        self.log_dir = '../runs/logs/'
        # print log every log-iter
        self.log_iter = 10
        # log every train-iter
        self.train_log_iter = 1000
        # skip validation during training
        self.skip_val = False
        # run validation every val-epoch
        self.val_epoch = 1

        # default settings for epochs, batch_size and lr
        self.lr = self.lr / 4 * self.batch_size

        # image transform for train
        self.transformForAll = pairedTr.Compose([
            pairedTr.RandomPerspective(distortion_scale=0.3, p=0.2),
            pairedTr.RandomResizedCrop(
                (self.crop_size_h, self.crop_size_w),
                scale=(0.75, 1.0),
                ratio=(self.crop_size_w/self.crop_size_h, self.crop_size_w/self.crop_size_h)),
        ])

        self.transformForImage = pairedTr.Compose([
            pairedTr.ColorJitter(0.3, 0.3, 0.3),
            pairedTr.ToTensor(),
            pairedTr.RandomErasing(p=0.2),
            pairedTr.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.transformForSeg = None

        # image transform for val

        self.transformForAll_val = pairedTr.Compose([
            pairedTr.RandomResizedCrop(
                (self.crop_size_h, self.crop_size_w),
                scale=(0.75, 1.0),
                ratio=(self.crop_size_w/self.crop_size_h, self.crop_size_w/self.crop_size_h)),
        ])

        self.transformForImage_val = pairedTr.Compose([
            pairedTr.ToTensor(),
            pairedTr.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.transformForSeg_val = None
        # dataset and dataloader
        data_kwargs = {'transformForAll': self.transformForAll,
                       'transformForImage': self.transformForImage,
                       'transformForSeg': self.transformForSeg,
                       'rootDir': self.rootDir,
                       'segDistinguishInstance': True}
        data_kwargs_val = {'transformForAll': self.transformForAll_val,
                           'transformForImage': self.transformForImage_val,
                           'transformForSeg': self.transformForSeg_val,
                           'rootDir': self.rootDir,
                           'segDistinguishInstance': True}

        self.trainset = get_segmentation_dataset(
            self.dataset, split='train', **data_kwargs)
        self.trainSetLen = len(self.trainset)

        self.iters_per_epoch = len(
            self.trainset) // self.batch_size
        self.max_iters = self.epochs * self.iters_per_epoch

        train_sampler = make_data_sampler(
            self.trainset, shuffle=True, distributed=False)
        train_batch_sampler = make_batch_data_sampler(
            train_sampler, self.batch_size, self.max_iters)
        self.train_loader = data.DataLoader(dataset=self.trainset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=self.workers,
                                            pin_memory=True)

        if not self.skip_val:
            self.valset = get_segmentation_dataset(
                self.dataset, split='val', **data_kwargs_val)
            val_sampler = make_data_sampler(
                self.valset, False, False)
            val_batch_sampler = make_batch_data_sampler(
                val_sampler, self.batch_size)
            self.val_loader = data.DataLoader(dataset=self.valset,
                                              batch_sampler=val_batch_sampler,
                                              num_workers=self.workers,
                                              pin_memory=True)

        self.model = get_segmentation_model(self.modelName)
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=self.max_iters,
                                         power=0.9,
                                         warmup_factor=self.warmup_factor,
                                         warmup_iters=self.warmup_iters,
                                         warmup_method=self.warmup_method)
