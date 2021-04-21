from __future__ import print_function

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable


import os
import glob
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.attention import Attention
import argparse
from preprocessing.create_bags import ICIARBags

image_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/breastcancer/data/breakhis/images'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True}

data=ICIARBags(image_path,n=100)
train_loader = torch.utils.data.DataLoader(data, shuffle = True, num_workers = 6, batch_size = 1)
#test_loader = torch.utils.data.DataLoader(val_data, shuffle = False, num_workers = 6, batch_size = 1)


print('Init Model')
model = Attention()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                       weight_decay=10e-5)
# optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    correct_label_pred = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # print("label",label[0][0])
        # print("label",label[0])
        bag_label = label[0]
        
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        data = data.squeeze(0)

        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        error, predicted_label_train = model.calculate_classification_error(data, bag_label)
        # print("bag_label,predicted_label_train",bag_label,predicted_label_train)
        # print(int(bag_label) == int(predicted_label_train))
        correct_label_pred += (int(bag_label) == int(predicted_label_train))
        # exit()
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        # print(correct_label_pred)
        # print(len(train_loader))
        # print(batch_idx)

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    train_acc = (1 - train_error)*100


    result_train = 'Epoch: {}, Loss: {:.4f}, Train error: {:.4f}, Train accuracy: {:.2f}'.format(epoch, train_loss.cpu().numpy()[0], train_error, train_acc)

    print(result_train)
    return result_train


for epoch in range(1, args.epochs + 1):
    print('----------Start Training----------')
    train_result = train(epoch)
    print('----------Start Testing----------')
