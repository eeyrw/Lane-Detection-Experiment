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


class Trainer(object):
    def __init__(self, args):
        self.exprHelper = ExperimentHelper.ExperimentHelper(args)
        self.device = self.exprHelper.device
        args = self.exprHelper.args

        # image transform for train
        transFormsForAll = pairedTr.Compose([
            pairedTr.RandomPerspective(distortion_scale=0.3, p=0.2),
            pairedTr.RandomResizedCrop(
                (args.crop_size_h, args.crop_size_w), scale=(0.75, 1.0), ratio=(args.crop_size_w/args.crop_size_h, args.crop_size_w/args.crop_size_h)),
        ])

        transFormsForImage = pairedTr.Compose([
            pairedTr.ColorJitter(0.3, 0.3, 0.3),
            pairedTr.ToTensor(),
            pairedTr.RandomErasing(p=0.2),
            pairedTr.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        transFormsForSeg = None

        # image transform for val

        transFormsForAll_val = pairedTr.Compose([
            pairedTr.RandomResizedCrop(
                (args.crop_size_h, args.crop_size_w), scale=(0.75, 1.0), ratio=(args.crop_size_w/args.crop_size_h, args.crop_size_w/args.crop_size_h)),
        ])

        transFormsForImage_val = pairedTr.Compose([
            pairedTr.ToTensor(),
            pairedTr.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        transFormsForSeg_val = None

        # dataset and dataloader
        data_kwargs = {'transformForAll': transFormsForAll,
                       'transformForImage': transFormsForImage,
                       'transformForSeg': transFormsForSeg,
                       'rootDir': args.rootDir}
        data_kwargs_val = {'transformForAll': transFormsForAll_val,
                           'transformForImage': transFormsForImage_val,
                           'transformForSeg': transFormsForSeg_val,
                           'rootDir': args.rootDir}

        trainset = get_segmentation_dataset(
            args.dataset, split='train', **data_kwargs)
        self.trainSetLen = len(trainset)

        train_sampler = make_data_sampler(
            trainset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(
            train_sampler, args.batch_size, args.max_iters)
        self.train_loader = data.DataLoader(dataset=trainset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        if not args.skip_val:
            valset = get_segmentation_dataset(
                args.dataset, split='val', **data_kwargs_val)
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

        self.criterion = torch.nn.BCEWithLogitsLoss(
            reduction='mean', pos_weight=torch.tensor([20])).to(self.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        self.best_pred = 0.0
        self.best_val_loss = 1000000

    def train(self):
        self.exprHelper.trainPrepare(self.trainSetLen)
        self.model.train()
        for iteration, (images, targets) in enumerate(self.train_loader):
            iteration += 1
            self.exprHelper.updateIteration(iteration)

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if self.exprHelper.isTimeToLog():
                self.exprHelper.logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration,
                        self.exprHelper.args.max_iters,
                        self.optimizer.param_groups[0]['lr'],
                        loss,
                        str(datetime.timedelta(seconds=int(
                            time.time() - self.exprHelper.start_time))),
                        self.exprHelper.getEstimatedTime()))

                self.exprHelper.writer.add_scalar(
                    'Loss/train', loss, iteration)
                self.exprHelper.writer.add_scalar('HyperParameter/learning_rate',
                                                  self.optimizer.param_groups[0]['lr'], iteration)

            if self.exprHelper.isTimeToSaveCheckPoint():
                self.exprHelper.save_checkpoint(
                    self.model, self.args, is_best=False)

            if self.exprHelper.isTimeToValidate():
                val_loss = self.validation(self.exprHelper.writer, iteration)
                self.exprHelper.writer.add_scalar(
                    'Loss/val', val_loss, iteration)
                self.model.train()

            if self.exprHelper.isTimeToCheckTrainResult():
                self.exprHelper.visualizeImageAndLabel('TrainSamples', iteration, images[0].cpu(
                ), targets[0].cpu(), torch.sigmoid(outputs[0]).cpu())
        self.exprHelper.trainFinish(self.model)

    def validation(self, writer, iteration):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        lossList = []
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model(image)
                loss = self.criterion(output, target)
                lossList.append(loss)
            self.exprHelper.logger.info("Sample: {:d}, loss: {:.8f}".format(
                i + 1, loss))

            if i == 0:
                self.visualizeImageAndLabel(writer, 'ValidationSamples', iteration, image[0].cpu(
                ), target[0].cpu(), torch.sigmoid(output[0]).cpu())

        new_val_loss = torch.Tensor(lossList).mean()
        if new_val_loss < self.best_val_loss:
            is_best = True
            self.best_val_loss = new_val_loss
        self.exprHelper.save_checkpoint(self.model, is_best)
        synchronize()
        return new_val_loss


if __name__ == '__main__':

    print(os.getcwd())
    args = getParameter()
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
    writer.close()
