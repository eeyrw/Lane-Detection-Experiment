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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib
import light.data.sync_transforms as pairedTr


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
    parser.add_argument('--model', type=str, default='erfnet_lstm',
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
    parser.add_argument('--resume', type=str, default='erfnet_culane.pth',
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

        self.transFormsForAll_val = pairedTr.Compose([
            pairedTr.RandomResizedCrop(
                (args.crop_size_h, args.crop_size_w), scale=(0.9, 1.0), ratio=(2/1, 2/1)),
        ])

        self.transFormsForImage_val = pairedTr.Compose([
            pairedTr.ToTensor(),
            pairedTr.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.transFormsForSeg_val = None

        data_kwargs = {'transformForAll': self.transFormsForAll_val,
                       'transformForImage': self.transFormsForImage_val,
                       'transformForSeg': self.transFormsForSeg_val,
                       'requireRawImage': True,
                       'rootDir': args.rootDir,
                       'mode': 'consecutive',
                       'framesGroupSize': 10
                       }
        valset = get_segmentation_dataset(
            args.dataset, split='test8_night', **data_kwargs)
        self.val_loader = data.DataLoader(dataset=valset,
                                          shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model(
            args.model, dataset=args.dataset, isValMode=True).to(self.device)
        self.model2 = get_segmentation_model(
            args.model, dataset=args.dataset, isValMode=True,isNoMemory=True).to(self.device)
        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(
                    args.resume, map_location=lambda storage, loc: storage))
                self.model2.load_state_dict(torch.load(
                    args.resume, map_location=lambda storage, loc: storage))                    

    def visualizeImageAndLabel(self, image, label, output):
        maxVal = torch.max(image)
        minVal = torch.min(image)
        imageNormalized = (image-minVal)/(maxVal-minVal)
        maxVal = torch.max(label)
        minVal = torch.min(label)
        labelNormalized = (label.float()-minVal)/(maxVal-minVal)
        maxVal = torch.max(output)
        minVal = torch.min(output)
        outputNormalized = (output.float()-minVal)/(maxVal-minVal)
        outputNormalized = outputNormalized.detach() > 0.9

        fig2 = plt.figure(constrained_layout=True, figsize=[9, 8], dpi=100)
        canvas = FigureCanvas(fig2)

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
        canvas.draw()       # draw the canvas, cache the renderer
        plt.close(fig2)

        buf = canvas.buffer_rgba()
        # ... convert to a NumPy array ...
        X = np.asarray(buf)[:, :, (2, 1, 0)]
        return X

    def eval(self):
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
            model2 = self.model2
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        model2.eval()
        for image, target, rawImageFile in self.val_loader:
            # N,SEQ_LEN,C,H,W
            N, SEQ_LEN, C, H, W = image.shape

            for i in range(N):
                for j in range(SEQ_LEN):
                    img = image[i][j].unsqueeze(0).to(self.device)
                    tgt = target[i][j].unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = model(img)
                        o2 = model2(img)
                        X = self.visualizeImageAndLabel(
                            img[0], tgt[0], torch.sigmoid(outputs[0]))

                        cv2.imshow('AAA', X)
                        X = self.visualizeImageAndLabel(
                            img[0], tgt[0], torch.sigmoid(o2[0]))

                        cv2.imshow('BBB', X)
                        k = cv2.waitKey(1) & 0xff
                        if k == 27:
                            break
        cv2.destroyWindow('AAA')
        cv2.destroyWindow('BBB')

    def evalOnSingleImage(self, imagePath):
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        imageFile = imagePath
        rawImageRgb = Image.open(imageFile).convert('RGB')

        imageRgb = self.transFormsForImage_val(
            self.transFormsForAll_val(rawImageRgb))
        image = torch.unsqueeze(imageRgb, 0).to(self.device)
        with torch.no_grad():
            outputs = model(image)
            X = self.visualizeImageAndLabel(image[0], torch.zeros_like(
                outputs[0]), torch.sigmoid(outputs[0]))
            cv2.imshow('AAA', X)
            k = cv2.waitKey(0) & 0xff
        cv2.destroyWindow('AAA')

    def evalOnVideo(self):
        videos = [r"C:\Users\yuan\Documents\研究生课题\LDW Research\highway45.mp4",
                  r"C:\Users\yuan\Documents\研究生课题\LDW Research\Mitsubishi Evo VIII MR - Forza Horizon 4 _ Logitech g29 gameplay.mp4",
                  r"C:\Users\yuan\Desktop\lane-detector-master\REC073.mp4",
                  r"C:\Users\yuan\Documents\研究生课题\drivingVideo.mp4",
                  r"C:\Users\yuan\Documents\研究生课题\pikes peak.mp4",
                  r"E:\Lane Dataset\Jiqing Expressway Video\IMG_0308.mov",
                  r"E:\Lane Dataset\Jiqing Expressway Video\IMG_0253.mov",
                  r"C:\Users\yuan\Desktop\QQ视频_c8a5486414df1c55787f097d6af685281590107727.mp4"
                  ]
        # The video feed is read in as a VideoCapture object

        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        cap = cv2.VideoCapture(videos[2])
        # out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('H','2','6','4'), 30, (800,900))
        while (cap.isOpened()):
            # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
            for i in range(1):
                ret, frame = cap.read()
            if frame is None:
                break
            imageRgb = self.transFormsForImage_val(self.transFormsForAll_val(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
            image = torch.unsqueeze(imageRgb, 0).to(self.device)
            with torch.no_grad():
                outputs = model(image)
                X = self.visualizeImageAndLabel(image[0], torch.zeros_like(
                    outputs[0]), torch.sigmoid(outputs[0]))
                cv2.imshow('AAA', X)
                # out.write(X)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # The following frees up resources and closes all windows
        cap.release()
        cv2.destroyAllWindows()

    def evalOnVideo2(self):
        videos = [r"C:\Users\yuan\Documents\研究生课题\LDW Research\highway45.mp4",
                  r"C:\Users\yuan\Documents\研究生课题\LDW Research\Mitsubishi Evo VIII MR - Forza Horizon 4 _ Logitech g29 gameplay.mp4",
                  r"C:\Users\yuan\Desktop\lane-detector-master\REC073.mp4",
                  r"C:\Users\yuan\Documents\研究生课题\drivingVideo.mp4",
                  r"C:\Users\yuan\Documents\研究生课题\pikes peak.mp4",
                  r"E:\Lane Dataset\Jiqing Expressway Video\IMG_0308.mov",
                  r"E:\Lane Dataset\Jiqing Expressway Video\IMG_0253.mov",
                  r"C:\Users\yuan\Desktop\QQ视频_c8a5486414df1c55787f097d6af685281590107727.mp4"
                  ]
        # The video feed is read in as a VideoCapture object

        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
            model2 = self.model2
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        model2.eval()
        cap = cv2.VideoCapture(videos[6])
        # out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('H','2','6','4'), 30, (800,900))
        while (cap.isOpened()):
            # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
            for i in range(10):
                ret, frame = cap.read()
            if frame is None:
                break
            imageRgb = self.transFormsForImage_val(self.transFormsForAll_val(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
            image = torch.unsqueeze(imageRgb, 0).to(self.device)
            with torch.no_grad():
                outputs = model(image)
                o2 = model2(image)
                X = self.visualizeImageAndLabel(image[0], torch.zeros_like(
                    outputs[0]), torch.sigmoid(outputs[0]))
                cv2.imshow('AAA', X)
                X = self.visualizeImageAndLabel(image[0], torch.zeros_like(
                    outputs[0]), torch.sigmoid(o2[0]))
                cv2.imshow('BBB', X)
                # out.write(X)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # The following frees up resources and closes all windows
        cap.release()
        cv2.destroyAllWindows()

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
    

    # args.resume = 'erfnet_culane_best_model_lowRes-2020.05.23.12.52.56.pth '
    args.resume = 'erfnet_lstm_culane_best_model_clstm_10frames_scratch-2020.05.29.08.34.15.pth'

    args.crop_size_h = 256
    args.crop_size_w = 512
    evaler = Evaler(args)
    # evaler.eval()
    # evaler.evalOnSingleImage(r"c:\Users\yuan\Desktop\0460080053.jpg")
    evaler.evalOnVideo2()
    torch.cuda.empty_cache()
