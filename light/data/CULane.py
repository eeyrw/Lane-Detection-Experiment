from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset


class CULaneDataset(Dataset):
    NUM_CLASS = 1

    splitDict = {
        'train': {'isTest': False, 'dir': 'list/train_gt.txt'},
        'val': {'isTest': False, 'dir': 'list/val_gt.txt'},
        'test': {'isTest': True, 'dir': 'list/test.txt'},
        'test0_normal': {'isTest': True, 'dir': 'list/test_split/test0_normal.txt'},
        'test1_crowd': {'isTest': True, 'dir': 'list/test_split/test1_crowd.txt'},
        'test2_hlight': {'isTest': True, 'dir': 'list/test_split/test2_hlight.txt'},
        'test3_shadow': {'isTest': True, 'dir': 'list/test_split/test3_shadow.txt'},
        'test4_noline': {'isTest': True, 'dir': 'list/test_split/test4_noline.txt'},
        'test5_arrow': {'isTest': True, 'dir': 'list/test_split/test5_arrow.txt'},
        'test6_curve': {'isTest': True, 'dir': 'list/test_split/test6_curve.txt'},
        'test7_cross': {'isTest': True, 'dir': 'list/test_split/test7_cross.txt'},
        'test8_night': {'isTest': True, 'dir': 'list/test_split/test8_night.txt'},
    }

    def __init__(self, rootDir, split='train', requireRawImage=False, transformForSeg=None, transformForImage=None):
        self.rootDir = rootDir
        self.transformForImage = transformForImage
        self.transformForSeg = transformForSeg
        self.requireRawImage = requireRawImage
        self.split = split

        self.filePairList = []
        self.indexFilePath = os.path.join(
            rootDir, self.splitDict[split]['dir'])
        self.isTest = self.splitDict[split]['isTest']

        with open(self.indexFilePath, 'r') as f:
            for line in f.readlines():
                if not self.isTest:
                    imagePath, segImagePath, _, _, _, _ = line.split()
                    if imagePath[0] == '/':
                        imagePath = imagePath[1:]
                    if segImagePath[0] == '/':
                        segImagePath = segImagePath[1:]
                else:
                    line = line.rstrip()
                    if line[0] == '/':
                        line = line[1:]

                    baseDir = os.path.dirname(line)
                    fileName = os.path.basename(line)
                    # Use splitext() to get filename and extension separately.
                    (fileNameWithoutExt, _) = os.path.splitext(fileName)
                    imagePath = line
                    segImagePath = os.path.join(
                        rootDir, 'laneseg_label_w16_test', baseDir, fileNameWithoutExt+'.png')

                self.filePairList.append(
                    (os.path.join(rootDir, imagePath), os.path.join(rootDir, segImagePath)))

    def __getitem__(self, idx):
        imageFile, segFile = self.filePairList[idx]
        rawImageRgb = Image.open(imageFile).convert('RGB')
        rawSegImage = np.clip(cv2.imread(
            segFile, cv2.IMREAD_UNCHANGED), 0, 1)
        if self.transformForImage is not None:
            imageRgb = self.transformForImage(rawImageRgb)
        else:
            imageRgb = transforms.ToTensor()(imageRgb)

        segImage = torch.squeeze(torch.from_numpy(rawSegImage)).long()
        if self.requireRawImage:
            return imageRgb, segImage, imageFile
        else:
            return imageRgb, segImage

    def __len__(self):
        return len(self.filePairList)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS
