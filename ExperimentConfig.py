import datetime
import argparse


class ExperimentConfig(object):
    def __init__(self):
        # experiment name (default: defaultExpr)
        self.exprName = 'defaultExpr'

        # model name (default: erfnet)
        self.model = 'erfnet'
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
