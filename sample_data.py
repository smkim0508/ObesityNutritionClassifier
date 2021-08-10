from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np

img = Image.open('/Users/sungmin/Desktop/Dataset/Train_samples/hamburger/images (10).jpeg')

padded_imgs = [T.Pad(padding=padding)(img) for padding in (3, 10, 30, 50)]
idx = 1
for pad_img in padded_imgs:
    pad_img.save('padded' + str(idx)+'.png')
    idx += 1

perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
perspective_imgs = [perspective_transformer(img) for _ in range(4)]
idx = 1
for perspective_img in perspective_imgs:
    perspective_img.save('perspective'+ str(idx) + '.png')
    idx +=1


affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
affine_imgs = [affine_transfomer(img) for _ in range(4)]
idx = 1
for affine_img in affine_imgs:
    affine_img.save('affine' + str(idx)+ '.png')
    idx += 1

hflipper = T.RandomHorizontalFlip(p=0.5)
transformed_imgs = [hflipper(img) for _ in range(4)]
idx = 1
for hflipped_img in transformed_imgs:
    hflipped_img.save('hflipped' + str(idx)+ '.png')
    idx += 1

vflipper = T.RandomVerticalFlip(p=0.5)
transformed_imgs = [vflipper(img) for _ in range(4)]
idx = 1
for vflipped_img in transformed_imgs:
    vflipped_img.save('vflipped' + str(idx) + '.png')
    idx +=1 

rotater = T.RandomRotation(degrees=(0, 180))
rotated_imgs = [rotater(img) for _ in range(4)]
idx = 1
for rotated_img in rotated_imgs:
    rotated_img.save('rotated' + str(idx) + '.png')
    idx += 1
