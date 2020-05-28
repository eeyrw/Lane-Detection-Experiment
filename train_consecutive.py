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
from DiceLoss import BatchSoftDiceLoss
from DiceLoss import BatchSoftBinaryDiceLoss
from torch.utils.tensorboard import SummaryWriter
import light.data.sync_transforms as pairedTr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.experimentName = args.exprName

        # image transform for train
        transFormsForAll = pairedTr.Compose([
            pairedTr.RandomPerspective(distortion_scale=0.2, p=0.1),
            pairedTr.RandomResizedCrop(
                (args.crop_size_h, args.crop_size_w), scale=(0.75, 1.0), ratio=(args.crop_size_w/args.crop_size_h, args.crop_size_w/args.crop_size_h)),
        ])

        transFormsForImage = pairedTr.Compose([
            pairedTr.ColorJitter(0.3, 0.3, 0.3),
            pairedTr.ToTensor(),
            pairedTr.RandomErasing(p=0.1),
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
                       'rootDir': args.rootDir,
                       'mode': 'consecutive',
                       'framesGroupSize': 10
                       }
        data_kwargs_val = {'transformForAll': transFormsForAll_val,
                           'transformForImage': transFormsForImage_val,
                           'transformForSeg': transFormsForSeg_val,
                           'rootDir': args.rootDir,
                           'mode': 'consecutive',
                           'framesGroupSize': 10}

        trainset = get_segmentation_dataset(
            args.dataset, split='train', **data_kwargs)
        args.iters_per_epoch = len(
            trainset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

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

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(
                    args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        # self.criterion = MixSoftmaxCrossEntropyLoss(
        #   args.aux, args.aux_weight, ignore_index=-1).to(self.device)
        # self.criterion = SoftDiceLoss().to(self.device)
        # self.criterion = BatchSoftDiceLoss().to(self.device)
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

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, args.step_size, gamma=0.1, last_epoch=-1)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[
                                                                 args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric = SegmentationMetric(trainset.num_class)

        self.best_pred = 0.0
        self.best_val_loss = 1000000

    def visualizeImageAndLabel(self, writer, tag, step, images, labels, outputs):
        # images [SEQ_LEN,C,H,W]
        imageLen = images.shape[0]
        labelLen = labels.shape[0]
        outputLen = outputs.shape[0]
        assert imageLen == labelLen and outputLen == labelLen
        mainFig = plt.figure(constrained_layout=True,
                             figsize=[9*imageLen, 8], dpi=100)
        spec = gridspec.GridSpec(ncols=imageLen, nrows=2, figure=mainFig)
        for i in range(imageLen):
            image = images[i]
            label = labels[i]
            output = outputs[i]
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

            f2_ax1 = mainFig.add_subplot(spec[0, i])
            f2_ax2 = mainFig.add_subplot(spec[1, i])

            imageNormalized = imageNormalized.permute(1, 2, 0)  # CHW to HWC

            f2_ax1.set_title("Predict-%d"%i)
            f2_ax1.imshow(imageNormalized, interpolation='bilinear')
            f2_ax1.imshow(outputNormalized[0], alpha=outputNormalized[0]
                          * 0.7, cmap=plt.cm.rainbow, vmin=0, vmax=1)
            f2_ax2.set_title("Ground Truth-%d"%i)
            f2_ax2.imshow(imageNormalized, interpolation='bilinear')
            f2_ax2.imshow(labelNormalized[0], alpha=labelNormalized[0]
                          * 0.7, cmap=plt.cm.rainbow, vmin=0, vmax=1)
        writer.add_figure(tag, mainFig,
                          global_step=step, close=True, walltime=None)

    def train(self):
        global logger
        self.experimentStartTime = time.strftime(
            '%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))
        logger = setup_logger(args.model, args.log_dir, get_rank(), filename='{}_{}_log_{}-{}.txt'.format(
            args.model, args.dataset, self.experimentName, self.experimentStartTime))

        logger.info("Using {} GPUs".format(num_gpus))
        logger.info(args)

        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * \
            self.args.iters_per_epoch

        checkDsPerIters = 100
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()

        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(
            epochs, max_iters))
        writer = SummaryWriter(comment='-%s-%s' %
                               (self.experimentName, self.experimentStartTime))

        self.model.train()
        for iteration, (images, targets) in enumerate(self.train_loader):
            iteration += 1

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            losses = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) /
                           iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

                writer.add_scalar('Loss/train', losses, iteration)
                writer.add_scalar('HyperParameter/learning_rate',
                                  self.optimizer.param_groups[0]['lr'], iteration)

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args,  self.experimentName,
                                self.experimentStartTime, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                val_loss = self.validation(writer, iteration)
                writer.add_scalar('Loss/val', val_loss, iteration)
                self.model.train()

            if iteration % checkDsPerIters == 0 or iteration == 1:
                self.visualizeImageAndLabel(writer, 'TrainSamples', iteration, images[0].cpu(
                ), targets[0].cpu(), torch.sigmoid(outputs[0]).cpu())

        save_checkpoint(self.model, self.args, self.experimentName,
                        self.experimentStartTime, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, writer, iteration):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        lossList = []
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model(image)
                losses = self.criterion(output, target)
                lossList.append(losses)
            # self.metric.update(outputs[0], target[0])
            # pixAcc, mIoU = self.metric.get()
            # logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            #    i + 1, pixAcc, mIoU))
            logger.info("Sample: {:d}, loss: {:.8f}".format(
                i + 1, losses))

            if i == 0:
                self.visualizeImageAndLabel(writer, 'ValidationSamples', iteration, image[0].cpu(
                ), target[0].cpu(), torch.sigmoid(output[0]).cpu())

        # new_pred = (pixAcc + mIoU) / 2
        new_val_loss = torch.Tensor(lossList).mean()
        if new_val_loss < self.best_val_loss:
            is_best = True
            self.best_val_loss = new_val_loss
        save_checkpoint(self.model, self.args, self.experimentName,
                        self.experimentStartTime, is_best)
        synchronize()
        return new_val_loss


def save_checkpoint(model, args, experimentName, experimentStartTime, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}-{}.pth'.format(args.model, args.dataset,
                                        experimentName, experimentStartTime)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_best_model_{}-{}.pth'.format(
            args.model, args.dataset, experimentName, experimentStartTime)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':

    print(os.getcwd())
    args = getParameter()
    args.model = 'erfnet_lstm'

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

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
    writer.close()
