from PIL import ImageFilter
import random
import torchvision.datasets as datasets

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

img_dir = "/Users/junru/Desktop/imagenet_shape_10/train_0"

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        shape_path = img_path.replace('train', 'shape')
        shape_path = shape_path.replace('JPEG', 'jpg')
        img_sample = self.loader(img_path)
        img_sample.show()
        shape_sample = self.loader(shape_path)
        if self.transform is not None:
            img_sample = self.transform(img_sample)
            shape_sample = self.transform(shape_sample)
        return img_sample, shape_sample

trainset = ImageFolderInstance(img_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(
        trainset, shuffle=False,batch_size=1)

for i, (img, shape) in enumerate(train_loader):
    shape = shape.squeeze(0)
    shape[shape < 0.5] = 0
    shape[shape >= 0.5] = 1
    T = transforms.ToPILImage()
    shape = T(shape)

    shape.show()
    if i == 15:
        break

