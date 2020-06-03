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



class ExperimentHelper(object):

    def __init__(self, args):
        self.args = args
        self.experimentName = args.exprName
        self.device = self._getDeviceUsage()

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

    def trainPrepare(self):
        self.experimentStartTime = time.strftime(
            '%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))
        self.logger = setup_logger(self.args.model, self.args.log_dir, 0, filename='{}_{}_log_{}-{}.txt'.format(
            self.args.model, self.args.dataset, self.experimentName, self.experimentStartTime))
        self.logger.info(self.args)

        epochs, max_iters = self.args.epochs, self.args.max_iters
        self.val_per_iters = self.args.val_epoch * self.args.iters_per_epoch

        checkDsPerIters = 100
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        self.start_time = time.time()

        self.logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(
            epochs, max_iters))
        self.writer = SummaryWriter(comment='-%s-%s' %
                                    (self.experimentName, self.experimentStartTime))

    def getEstimatedTime(self, iteration):
        eta_seconds = ((time.time() - self.start_time) /
                       iteration) * (self.args.max_iters - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        return eta_string

    def trainFinish(self, model):
        self.save_checkpoint(model, is_best=False)
        total_training_time = time.time() - self.start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / self.args.max_iters))
