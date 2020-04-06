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

class CULaneDataset(Dataset):
    NUM_CLASS = 2
    def __init__(self, rootDir, split='train', mode='train', transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.split = split

        self.trainIndexFilePath = os.path.join(rootDir, r'list/train_gt.txt')
        self.trainFilePairList = []

        with open(self.trainIndexFilePath, 'r') as f:
            for line in f.readlines():
                imagePath, segImagePath, lane0, lane1, lane2, lane4 = line.split()
                self.trainFilePairList.append(
                    (os.path.join(rootDir, imagePath[1:]), os.path.join(rootDir, segImagePath[1:]), lane0, lane1, lane2, lane4))

    def __getitem__(self, idx):
        imageFile, segFile, _, _, _, _ = self.trainFilePairList[idx]
        img_rgb = Image.open(imageFile).convert('RGB')
        segImage = np.clip(cv2.imread(segFile, cv2.IMREAD_UNCHANGED), 0, 1)
        # segImageBackground = np.ones_like(segImage)-segImage
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            t=transforms.ToTensor()
            segImage = torch.squeeze(t(segImage)).long()
            # segImageBackground= t(segImageBackground)
            # segImages=t(np.stack((segImage,segImageBackground),axis=2))
        return img_rgb, segImage

    def __len__(self):
        return len(self.trainFilePairList)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS    
