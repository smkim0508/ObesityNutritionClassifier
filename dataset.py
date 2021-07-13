import os, sys
import numpy as np
import cv2
'''
import torch
from torch.utils.data import Dataset, Dataloader
import torchvision
import torchvision.transforms as transforms
'''
#/Users/sungmin/Desktop/Dataset

class JunkFoodSet: #(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.gt = [] #ground truth

        data_list = os.listdir(self.root)

        self.class_index = {'bacon':0, 'buffalo_wing': 1, 'cake': 2, 
        'cereal': 3, 'chicken_nuggets': 4, 'churros': 5, 'cinnamon_roll': 6,
        'cookie': 7, 'corn_dog': 8, 'cotton_candy': 9, 'cupcake': 10, 
        'donut': 11, 'french_fries': 12, 'hamburger': 13, 'hot_dog': 14,
        'icecream': 15, 'jell-o': 16, 'milk_shake': 17, 'pizza': 18,
        'poptarts': 19, 'soda': 20}
        for dir_name in data_list:
            category_dir = os.path.join(self.root, dir_name)
            if dir_name == '.DS_Store': 
                continue
            for img_name in os.listdir(category_dir):
                img_full_path = os.path.join(category_dir, img_name)
                self.samples.append(img_full_path)
                self.class_index[dir_name.lower()]
                self.gt.append(self.class_index[dir_name.lower()])

        if len(self.samples) != len(self.gt):
            print('error occurred!')
            exit(1)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        # 1. image read (using cv2) H x W x C, 0~255
        img = cv2.imread(self.samples[index])
        
        if self.transform:
            img = self.transform(img) # H x W x C => C x H x W & 0 ~ 255 => 0 ~ 1 & -1 ~ 1

        
        # 2. one-hot vector
        return img
        # C x H x W, 0~1 -> -1 ~ 1
        # data augmentation 
if __name__ == '__main__':
    train_set = JunkFoodSet('/Users/sungmin/Desktop/Dataset')
    print(len(train_set))

    for data in train_set:
        print(data)
        exit(1)


        

