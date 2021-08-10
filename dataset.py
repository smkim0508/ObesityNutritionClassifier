import os, sys
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset #, Dataloader
import torchvision
import torchvision.transforms as transforms

class JunkFoodSet(Dataset):
    def __init__(self, root, train=True, transform=None, flip=True, rotate=True, crop=True):
        self.root = root
        self.transform = transform
        self.samples = []
        self.gt = [] #ground truth
        self.train = train

        if flip:
            self.flip = transforms.RandomHorizontalFlip(p=.5)
        else:
            self.flip = None
            
        if rotate:
            self.rotate = transforms.RandomRotation(degrees=(0,180))
        else:
            self.rotate = None 
        '''
        if crop:
            self.crop = transforms.RandomCrop(size=(512, 512))
        else:
            self.crop = None
        '''
        self.resize = transforms.Resize(size=(512,512))
        # 119, 159   1736, 1200

        data_list = os.listdir(self.root)
       
        self.class_index = {'bacon':0, 'buffalo_wing': 1, 'cake': 2, 
        'cereal': 3, 'chicken_nuggets': 4, 'churros': 5, 'cinnamon_roll': 6,
        'cookie': 7, 'corn_dog': 8, 'cotton_candy': 9, 'cupcake': 10, 
        'donut': 11, 'french_fries': 12, 'hamburger': 13, 'hot_dog': 14,
        'icecream': 15, 'jell-o': 16, 'milk_shake': 17, 'pizza': 18,
        'poptarts': 19, 'soda': 20}

        self.num_class = len(self.class_index.keys())

        for dir_name in data_list:
            category_dir = os.path.join(self.root, dir_name)
            if dir_name == '.DS_Store': 
                continue
            for img_name in os.listdir(category_dir):
                if img_name == '.DS_Store':
                    continue
                if img_name.split(".")[-1] != "jpeg":
                    continue
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
        #img = cv2.imread(self.samples[index])
        class_name = self.samples[index].split('/')[-2]

        img = Image.open(self.samples[index])
        img = self.resize(img) # resize to 512 x 512
        
        # random horizontal flip
        if self.flip:
            img = self.flip(img)

        # random rotate
        if self.rotate:
            if np.random.rand() < 0.5:
                img = self.rotate(img)
        
        # whitening
        if self.transform:
            img = self.transform(img)
       
        #gt = torch.zeros(self.num_class, dtype=torch.float32)
        #gt[self.class_index[class_name]] = 1.

        gt = self.class_index[class_name]
        #! one hot vector         
        return (img, gt)
        

def tensor2numpy(img):
    img = ((img*0.5) + 0.5).clamp(0.0, 1.0) # -1~1 -> 0 ~ 1
    # 0 ~ 1 -> 0 ~ 255  
    np_img = (img.cpu().detach() * 255.).numpy().astype(np.uint8)
    # C x H x W -> H x W x C
    np_img = np_img.transpose(1,2,0)[:,:,::-1]
    return np_img

if __name__ == '__main__':
    junkfood_transform = transforms.Compose(
        [
            transforms.ToTensor(), # 0~255 -> 0~1
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # 0~1 -> -1~1
        ])
    train_set = JunkFoodSet('./Dataset', transform=junkfood_transform, 
                            flip=False, rotate=True, crop=False)
 

    for idx, data in enumerate(train_set):
        if idx == 5:
            break
        
        img = tensor2numpy(data)
        
        file_path = str(idx+1)+'.png'
        cv2.imwrite(file_path, img)