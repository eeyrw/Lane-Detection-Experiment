from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset


class CULaneDataset(Dataset):
    NUM_CLASS=1
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

    def __init__(self, rootDir,
                 split='train',
                 mode='discrete',
                 framesGroupSize=2,
                 requireRawImage=False,
                 transformForAll=None,
                 transformForSeg=None,
                 transformForImage=None,
                 resizeAndCropTo=(-1, -1),
                 segDistinguishInstance=False,
                 ):
        self.rootDir = rootDir
        self.transformForAll = transformForAll
        self.transformForImage = transformForImage
        self.transformForSeg = transformForSeg
        self.requireRawImage = requireRawImage
        self.split = split
        self.mode = mode
        self.framesGroupSize = framesGroupSize
        self.dataSetLen = 0
        self.wantedWidth = resizeAndCropTo[0]
        self.wantedHeight = resizeAndCropTo[1]
        self.segDistinguishInstance = segDistinguishInstance

        if self.wantedWidth > 0 and self.wantedHeight > 0:
            self.doResizeAndCrop = True
        else:
            self.doResizeAndCrop = False

        if self.segDistinguishInstance:
            self.NUM_CLASS = 5
        else:
            self.NUM_CLASS = 1

        self.filePairList = []
        self.indexFilePath = os.path.join(
            rootDir, self.splitDict[split]['dir'])
        self.isTest = self.splitDict[split]['isTest']

        rawframesGroupDict = {}

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

                if self.mode == 'discrete':
                    self.filePairList.append(
                        (os.path.join(rootDir, imagePath), os.path.join(rootDir, segImagePath)))
                else:
                    framesGroupName = os.path.basename(
                        os.path.dirname(imagePath))
                    rawframesGroupDict.setdefault(framesGroupName, []).append(
                        (os.path.join(rootDir, imagePath), os.path.join(rootDir, segImagePath)))

            if self.mode == 'discrete':
                self.dataSetLen = len(self.filePairList)
            else:
                self.framesGroupList = self._sliceFramesGroup(
                    rawframesGroupDict)
                self.dataSetLen = len(self.framesGroupList)

    def __getitem__(self, idx):
        if self.mode == 'discrete':
            return self._filePairToTensor(self.filePairList[idx], self.requireRawImage)
        else:
            imagesRgb = []
            segImages = []
            imageFiles = []
            for filePair in self.framesGroupList[idx]:
                if self.requireRawImage:
                    imageRgb, segImage, imageFile = self._filePairToTensor(
                        filePair, self.requireRawImage)
                    imageFiles.append(imageFile)
                else:
                    imageRgb, segImage = self._filePairToTensor(
                        filePair, self.requireRawImage)
                imagesRgb.append(imageRgb)
                segImages.append(segImage)
            if self.requireRawImage:
                return torch.stack(imagesRgb), torch.stack(segImages), torch.stack(imageFiles)
            else:
                return torch.stack(imagesRgb), torch.stack(segImages)

    def __len__(self):
        return self.dataSetLen

    @classmethod
    def _resizeAndCropToTargetSize(self, img, width, height):
        rawW, rawH = img.size
        rawAspectRatio = rawW/rawH
        wantedAspectRatio = width/height
        if rawAspectRatio > wantedAspectRatio:
            scaleFactor = height/rawH
            widthBeforeCrop = int(rawW*scaleFactor)
            return img.resize((widthBeforeCrop, height), Image.BILINEAR). \
                crop(((widthBeforeCrop-width)//2, 0,
                      (widthBeforeCrop-width)//2+width, height))
        else:
            scaleFactor = width/rawW
            heightBeforeCrop = int(rawH*scaleFactor)
            return img.resize((width, heightBeforeCrop), Image.BILINEAR). \
                crop((0, (heightBeforeCrop-height)//2, width,
                      (heightBeforeCrop-height)//2+height))

    def _filePairToTensor(self, filePair, requireRawImage):
        imageFile, segFile = filePair
        rawImageRgb = Image.open(imageFile).convert('RGB')
        rawSegImage = Image.open(segFile).convert('L')

        if self.doResizeAndCrop:
            rawImageRgb = self._resizeAndCropToTargetSize(
                rawImageRgb, self.wantedWidth, self.wantedHeight)
            rawSegImage = self._resizeAndCropToTargetSize(
                rawSegImage, self.wantedWidth, self.wantedHeight)

        if self.transformForAll is not None:
            rawImageRgb, rawSegImage = self.transformForAll(
                rawImageRgb, rawSegImage)

        if self.transformForImage is not None:
            imageRgb = self.transformForImage(rawImageRgb)
        else:
            imageRgb = transforms.ToTensor()(rawImageRgb)

        if self.transformForSeg is not None:
            segImage = self.transformForSeg(rawSegImage)

        if not self.segDistinguishInstance:
            rawSegImage = np.clip(rawSegImage, 0, 1)
            segImage = torch.from_numpy(rawSegImage).unsqueeze(0).float()
        else:
            segImage =torch.from_numpy(np.array(rawSegImage)).unsqueeze(0).float()

        if requireRawImage:
            return imageRgb, segImage, imageFile
        else:
            return imageRgb, segImage

    def _sliceFramesGroup(self, rawframesGroupDict):
        framesGroupList = []
        for _, clips in rawframesGroupDict.items():
            clipNum = len(clips)
            for i in range(0, clipNum, self.framesGroupSize):
                if i + self.framesGroupSize <= len(clips):
                    framesGroupList.append(clips[i:i + self.framesGroupSize])
                else:
                    framesGroupList.append(clips[len(clips)-self.framesGroupSize:]) # Ensure every group have same length
        return framesGroupList

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS


# culaneDs = CULaneDataset('E:\CULane', split='test1_crowd', mode='consecutive')
# b = culaneDs[100]
# a = culaneDs[10]
