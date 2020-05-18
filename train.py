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


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform for train
        transFormsForAll = pairedTr.Compose([
            pairedTr.RandomPerspective(distortion_scale=0.3, p=0.2),
            pairedTr.RandomResizedCrop(
                (256, 512), scale=(0.6, 1.0), ratio=(2/1, 2/1)),
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
                (256, 512), scale=(0.6, 1.0), ratio=(2/1, 2/1)),
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
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=20).to(self.device)

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

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[
                                                                 args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric = SegmentationMetric(trainset.num_class)

        self.best_pred = 0.0

    def visualizeImageAndLabel(self, writer, image, label, output):
        maxVal = torch.max(image)
        minVal = torch.min(image)
        imageNormalized = (image-minVal)/(maxVal-minVal)
        maxVal = torch.max(label)
        minVal = torch.min(label)
        labelNormalized = (label.float()-minVal)/(maxVal-minVal)
        maxVal = torch.max(output)
        minVal = torch.min(output)
        outputNormalized = (output.float()-minVal)/(maxVal-minVal)
        writer.add_image('DsInspect/In',imageNormalized, 0, dataformats='CHW')
        writer.add_images('DsInspect/Label_Out', torch.stack((labelNormalized,outputNormalized)), 0, dataformats='NCHW')
        # writer.add_image('DsInspect/Out', outputNormalized.unsqueeze(0), 0, dataformats='CHW')


    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * \
            self.args.iters_per_epoch

        checkDsPerIters = 1000
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(
            epochs, max_iters))
        writer = SummaryWriter()

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
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                mIoU = self.validation()
                writer.add_scalar('mIoU/val', mIoU, iteration)
                self.model.train()

            if iteration % checkDsPerIters == 0 or iteration == 1:
                self.visualizeImageAndLabel(writer, images[0], targets[0],torch.sigmoid(outputs[0]))

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc, mIoU))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)
        synchronize()
        return mIoU


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format(args.model, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':

    print(os.getcwd())
    args = getParameter()

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

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
    writer.close()
