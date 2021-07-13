from dataset import JunkFoodSet
import os, sys
import argparse
import cv2 
import numpy as np
import json

import logging
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from tensorboardX import SummaryWriter

from model import SimpleNet
def main():
    '''
    cifar10_transform = transforms.Compose(
        [
        transforms.ToTensor(), #0 ~ 255 => 0 ~ 1.0
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
    )   
    '''
    train_set = JunkFoodSet('/Users/sungmin/Desktop/Dataset')
    # 0 dataset
    # 1 model
    model = SimpleNet()
    # 2 loss function
    # 3 train loop
    #   3-1 test
    #   3-2 save



if __name__ == '__main__':
    main()