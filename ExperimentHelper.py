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

import shutil
import datetime
import argparse
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time
from torch.utils.tensorboard import SummaryWriter
import light.data.sync_transforms as pairedTr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib
import importlib


class ExperimentHelper(object):

    def __init__(self, cfgPyFile):
        cfg = importlib.import_module(cfgPyFile)
        self.args = cfg.ExperimentConfig()
        self.experimentName = self.args.exprName
        self.device = self._getDeviceUsage()
        self.args.model.to(self.device)
        self.iteration = 0

    def _getDeviceUsage(self):
        if self.args.cuda_usage and torch.cuda.is_available():
            cudnn.benchmark = True
            device = "cuda"
        else:
            device = "cpu"
        return device

    def save_checkpoint(self, model, is_best=False):
        """Save Checkpoint"""
        directory = os.path.expanduser(self.args.save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = '{}_{}_{}-{}.pth'.format(self.args.model, self.args.dataset,
                                            self.experimentName, self.experimentStartTime)
        filename = os.path.join(directory, filename)

        torch.save(model.state_dict(), filename)
        if is_best:
            best_filename = '{}_{}_best_model_{}-{}.pth'.format(
                self.args.model, self.args.dataset, self.experimentName, self.experimentStartTime)
            best_filename = os.path.join(directory, best_filename)
            shutil.copyfile(filename, best_filename)

    def loadStateDict(self):
        # resume checkpoint if needed
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                name, ext = os.path.splitext(self.args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(self.args.resume))
                self.model.load_state_dict(torch.load(
                    self.args.resume, map_location=lambda storage, loc: storage))

    def trainPrepare(self, trainSetLen):
        self.start_time = time.time()
        self.experimentStartTime = time.strftime(
            '%Y.%m.%d.%H.%M.%S', time.localtime(self.start_time))

        self.args.iters_per_epoch = trainSetLen // self.args.batch_size
        self.args.max_iters = self.args.epochs * self.args.iters_per_epoch

        epochs, max_iters = self.args.epochs, self.args.max_iters
        self.val_per_iters = self.args.val_epoch * self.args.iters_per_epoch

        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch

        self.logger = setup_logger(self.args.modelName, self.args.log_dir, 0, filename='{}_{}_log_{}-{}.txt'.format(
            self.args.modelName, self.args.dataset, self.experimentName, self.experimentStartTime))
        self.writer = SummaryWriter(comment='-%s-%s' %
                                    (self.experimentName, self.experimentStartTime))
        self.logger.info(self.args)
        self.logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(
            epochs, max_iters))

    def updateIteration(self, iteration):
        self.iteration = iteration

    def getEstimatedTime(self):
        eta_seconds = ((time.time() - self.start_time) /
                       self.iteration) * (self.args.max_iters - self.iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        return eta_string

    def isTimeToLog(self):
        return self.iteration % self.args.log_iter == 0

    def isTimeToCheckTrainResult(self):
        return self.iteration % self.args.train_log_iter == 0 or self.iteration == 1

    def isTimeToSaveCheckPoint(self):
        return self.iteration % self.args.save_per_iters == 0

    def isTimeToValidate(self):
        return not self.args.skip_val and self.iteration % self.args.val_per_iters == 0

    def trainFinish(self, model):
        self.save_checkpoint(model, is_best=False)
        total_training_time = time.time() - self.start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / self.args.max_iters))

    def visualizeImageAndLabel(self, tag, step, image, label, output):
        maxVal = torch.max(image)
        minVal = torch.min(image)
        imageNormalized = (image-minVal)/(maxVal-minVal)
        maxVal = torch.max(label)
        minVal = torch.min(label)
        labelNormalized = (label.float()-minVal)/(maxVal-minVal)
        maxVal = torch.max(output)
        minVal = torch.min(output)
        outputNormalized = (output.float()-minVal)/(maxVal-minVal)
        outputNormalized = outputNormalized.detach()
        # writer.add_image('DsInspect/In',imageNormalized, 0, dataformats='CHW')
        # writer.add_images('DsInspect/Label_Out', torch.stack((labelNormalized,outputNormalized)), 0, dataformats='NCHW')
        # writer.add_image('DsInspect/Out', outputNormalized.unsqueeze(0), 0, dataformats='CHW')
        fig2 = plt.figure(constrained_layout=True, figsize=[9, 8], dpi=100)

        spec2 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax2 = fig2.add_subplot(spec2[1, 0])

        imageNormalized = imageNormalized.permute(1, 2, 0)  # CHW to HWC

        f2_ax1.set_title("Predict")
        f2_ax1.imshow(imageNormalized, interpolation='bilinear')
        f2_ax1.imshow(outputNormalized[0], alpha=outputNormalized[0]
                      * 0.7, cmap=plt.cm.rainbow, vmin=0, vmax=1)
        f2_ax2.set_title("Ground Truth")
        f2_ax2.imshow(imageNormalized, interpolation='bilinear')
        f2_ax2.imshow(labelNormalized[0], alpha=labelNormalized[0]
                      * 0.7, cmap=plt.cm.rainbow, vmin=0, vmax=1)
        self.writer.add_figure(tag, fig2,
                               global_step=step, close=True, walltime=None)
        self.writer.flush()
