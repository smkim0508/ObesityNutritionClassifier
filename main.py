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

from tensorboardX import SummaryWriter
from model import SimpleNet

def opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='num workers')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    parser.add_argument('--log_dir', type=str, default='./work_dir', help='path to log directory')
    parser.add_argument('--resume_path', type=str, default=None, help='path to resume model')
    return parser.parse_args()

def train(device, model, optimizer, criterion, scheduler, data_loader, data_set, writer, epoch):
    model = model.to(device)
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    
    for idx, data in enumerate(data_loader, 0):
        train_sample, ground_truth = data
        train_sample = train_sample.to(device)
        ground_truth = ground_truth.to(device)
        #set gradient to zero
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward pass
            logits = model(train_sample)
            ik, preds = torch.max(logits, 1)
            # calculate loss
            loss = criterion(logits, ground_truth)
            # backward pass
            loss.backward()
            # update gradient
            optimizer.step()

        # running loss and accuracy
        # stats
        running_loss += loss.item()
        running_corrects += torch.sum(preds == ground_truth.data)
        writer.add_scalar('loss/ce_loss', loss.item(), epoch*len(data_set) + idx)

        # print loss every step that is a multiple of 5 
        if idx % 5 == 4:
            print("{} - [Epoch-{}][{}/{}] loss: {:.2f}".format(
                datetime.now(), epoch, idx, len(data_loader), running_loss))
            running_loss = 0.0
        
    # single epoch (train or validation) end
    scheduler.step()

    # print epoch accuracy
    epoch_acc = running_corrects.double() / len(data_set)
    print("{0} - [{1}-Epoch-{2}] Acc: {3:.2f} , LR: {4:.5f}".format(
        datetime.now(), 'Train', epoch, epoch_acc*100.0, optimizer.param_groups[0]['lr']
    ))
    writer.add_scalar('acc/train', epoch_acc*100.0, epoch*len(data_set))


def main():
    args = opt()

    if not os.path.exists(os.path.join(args.log_dir, 'ckpt')):
        os.makedirs(os.path.join(args.log_dir, 'ckpt'))
    writer = SummaryWriter(args.log_dir)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        print('cannot use cuda')
   

    JunkFoodSet_transform = transforms.Compose(
        [
        transforms.ToTensor(), #0 ~ 255 => 0 ~ 1.0
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
    )   
  
    train_set = JunkFoodSet('/Users/sungmin/Desktop/Dataset', transform=JunkFoodSet_transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # 0 dataset
    # 1 model
    model = SimpleNet()
    # 2 loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    else:
        raise NotImplementedError

    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones = [80, 160], gamma=0.1)
    
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume.path) #load file

        model.load_state_dict(checkpoint['model'])  #load model
        optimizer.load_state_dict(checkpoint['optimizer']) #load optimizer
        scheduler.load_state_dict(checkpoint['scheduler']) #load scheduler

        model = model.to(device)

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        print("{} - loaded checkpoint from {}, start epoch: {}".format(
            datetime.now(), args.resume_path, checkpoint['epoch']))

        start_epoch = checkpoint['epoch']
        end_epoch = args.max_epoch     
    else:
        start_epoch = 1
        end_epoch = args.max_epoch

    model = model.to(device)
    best_acc = 0.0
    for epoch in range(start_epoch, end_epoch+1):
        #train for single epoch
        train(device, model, optimizer, criterion, scheduler, trainloader, train_set, writer, epoch)

        #save model every 1 epoch
        if epoch % 20 == 0:
            save_model_path = os.path.join(os.path.join(args.log_dir, 'ckpt'), str(epoch) + '.pth')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                }, save_model_path)

    #save best model
    save_model_path = os.path.join(os.path.join(args.log_dir, 'ckpt'), 'best.pth')
    model.load_state_dict(best_model)

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': args.max_epoch
    }, save_model_path)

    print('Finished Training!')
    # 3 train loop
    #   3-1 test
    #   3-2 save



if __name__ == '__main__':
    main()

