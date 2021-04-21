import os
import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.utils import Progbar
from tqdm import tqdm, trange

#import models.gated_attention
import models.attention


class Trainer():
    def __init__(self,model,device,optimizer,train_loader,valid_loader,
                 epochs):
        
        self.model=model
        self.device=device
        self.optimizer=optimizer
        self.train_loader=train_loader
        self.valid_loader=valid_loader
        self.epochs=epochs
        self.train_loss=0
        self.valid_loss=0
        
        
    def run(self):
        #progress_bar=trange(self.epochs,desc='progress')
        self.model.to(self.device)
        for i in range(self.epochs-1):
            train_loss,train_error=self._train()
            #valid_loss,valid_error=self._validation()    
            train_info=' Epoch:{}/{},trainloss-{:.2f},acc-{:.2f} '
            #valid_info='validloss-{:.2f},validerror-{:.2f}'
            #loss_temp=train_loss.detach()
            loss_temp=train_loss.cpu().numpy()[0]
            error_temp=train_error
            acc = (1-error_temp)*100
            #error_temp=train_error.detach() 
            #error_temp=error_temp.cpu().numpy()[0]
            print(train_info.format(i+1,self.epochs,loss_temp,acc))
            #print(valid_info.format(i+1,valid_loss,valid_error),end="")

        return self.train_loss #self.valid_loss
        
        
    def _train(self):
        self.model.train()
        train_loss=0.
        train_error=0.            
        prog = Progbar(len(self.train_loader))
        for i, (x,y) in enumerate(self.train_loader):
            x=x.float()
            y=y[0]
            x=x.to(self.device)
            y=y.to(self.device)
            self.optimizer.zero_grad()
            loss, _ = self.model.calculate_objective(x, y)
            train_loss += loss.data[0]
            error, _ = self.model.calculate_classification_error(x, y)
            train_error += error
            loss.backward()
            #for name, param in self.model.named_parameters():
                #print(name, param.grad)

            self.optimizer.step()
            prog.update(i)
        
        train_loss /= len(self.train_loader)
        train_error /= len(self.train_loader)
        return train_loss, train_error
        

    def _validation(self):
        self.model.eval()
        valid_loss=0
        valid_error=0
        for i, (x,y) in self.valid_loader:
            x=x.float()
            y=y[0]
            x=x.to(self.device)
            y=y.to(self.device)
            with torch.no_grad():
                loss, _ = model.calculate_objective(x, y)
                valid_loss += loss
                error, _ = model.calculate_classification_error(x, y)
                valid_error += error
            
        valid_loss /= len(valid_loader)
        valid_error /= len(valid_loader)
        return valid_loss,valid_error
                          
