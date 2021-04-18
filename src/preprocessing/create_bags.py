import os 
import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class ICIARBags(Dataset):
    def __init__(self,patch_path,labels,n=200,key={},name='labels'):
        self.n_size=n
        self.name=name
        self.key=key
        self.patch_paths=glob.glob(os.path.join(patch_path,'*'))
        self.patch_num=len(self.patch_paths)
        self.patch_bags,self.label_bags=self._generate_bags()
        self.positive_num=len(list(filter(lambda x: x==1, self.label_bags)))
        self.negative_num=len(list(filter(lambda x: x==0, self.label_bags)))
        
 
    def _generate_bags(self):
        
        train_patch_bags=[]
        train_label_bags=[]
        image_paths=glob.glob(os.path.join(self.patch_paths,'*'))
        for i, path in enumerate(image_paths):
            class_paths=glob.glob(os.path.join(path,'*'))
            train_label_bags.append(i)
            for c in class_paths:
                image=cv2.imread(c)
                patches=[image[i:i+28,j:j+28,:] for i in
                        range(0,28*16,28) for j in range(0,28*25,28)])
                train_patch_bags.append(torch.tensor(patches))

        return train_patch_bags, train_label_bags


    def __len__(self):
        return len(self.patch_bags)
    
   
    @staticmethod
    def _buildbag(patches):
        patches=torch.tensor([cv2.imread(p) for p in patches])
        patches=patches.permute(0,3,1,2)
      
        return patches
        
    
    def __getitem__(self,index):
        bag=self.patch_bags[index]
        bag=self._buildbag(bag)
        label=self.label_bags[index]
        
        return bag,label
