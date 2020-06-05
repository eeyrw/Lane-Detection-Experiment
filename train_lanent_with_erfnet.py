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
    def __init__(self, configPyFile):
        self.exprHelper = ExperimentHelper.ExperimentHelper(configPyFile)
        self.device = self.exprHelper.device
        self.args = self.exprHelper.args
        assert self.args.modelName == 'lanenet_erfnet'
        self.model = self.args.model
        self.optimizer = self.args.optimizer
        self.lr_scheduler = self.args.lr_scheduler

        self.best_pred = 0.0
        self.best_val_loss = 1000000

    def train(self):
        self.exprHelper.trainPrepare(self.args.trainSetLen)
        self.args.model.train()
        for iteration, (images, targets) in enumerate(self.args.train_loader):
            iteration += 1
            self.exprHelper.updateIteration(iteration)

            images = images.to(self.device)
            targets = targets.to(self.device)

            resultDict = self.args.model(images, targets)
            embedding = resultDict['embedding']
            binary_seg = resultDict['binary_seg']
            loss_seg = resultDict['loss_seg']
            loss_var = resultDict['loss_var']
            loss_dist = resultDict['loss_dist']
            reg_loss = resultDict['reg_loss']
            loss = resultDict['loss']

            self.args.optimizer.zero_grad()
            loss.backward()
            self.args.optimizer.step()
            self.args.lr_scheduler.step()

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
                    'Loss/trainTotal', loss, iteration)
                self.exprHelper.writer.add_scalar(
                    'Loss/trainSegmentation', loss_seg, iteration)
                self.exprHelper.writer.add_scalar(
                    'Loss/trainDistance', loss_dist, iteration)
                self.exprHelper.writer.add_scalar(
                    'Loss/trainVariance', loss_var, iteration)
                self.exprHelper.writer.add_scalar(
                    'Loss/trainReg', reg_loss, iteration)

                self.exprHelper.writer.add_scalar('HyperParameter/learning_rate',
                                                  self.args.optimizer.param_groups[0]['lr'], iteration)

            if self.exprHelper.isTimeToSaveCheckPoint():
                self.exprHelper.save_checkpoint(self.model, is_best=False)

            if self.exprHelper.isTimeToValidate():
                val_loss = self.validation(self.exprHelper.writer, iteration)
                self.exprHelper.writer.add_scalar(
                    'Loss/val', val_loss, iteration)
                self.args.model.train()

            if self.exprHelper.isTimeToCheckTrainResult():
                self.exprHelper.visualizeImageAndLabel('TrainSamples', iteration, images[0].cpu(
                ), targets[0].unsqueeze(0).cpu(), torch.sigmoid(binary_seg[0][0].unsqueeze(0)).cpu())
        self.exprHelper.trainFinish(self.model)

    def validation(self, writer, iteration):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        lossList = []
        for i, (image, target) in enumerate(self.args.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model(image)
                resultDict = self.model(image, target)
                embedding = resultDict['embedding']
                binary_seg = resultDict['binary_seg']
                loss = resultDict['loss']
                lossList.append(loss)
            self.exprHelper.logger.info("Sample: {:d}, loss: {:.8f}".format(
                i + 1, loss))

            if i == 0:
                self.exprHelper.visualizeImageAndLabel(writer, 'ValidationSamples', iteration, image[0].cpu(
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
    trainer = Trainer('ExperimentConfigERFNetLaneNet')
    trainer.train()
    torch.cuda.empty_cache()
    writer.close()
