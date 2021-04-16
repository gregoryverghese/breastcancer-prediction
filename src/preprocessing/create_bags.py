import os 
import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class PatchBags(Dataset):
    def __init__(self,patch_path,labels,n=200,key={},name='labels'):
        self.n_size=n
        self.name=name
        self.key=key
        self.patch_paths=glob.glob(os.path.join(patch_path,'*'))
        self.labels=labels
        self.patch_num=len(self.patch_paths)
        self.patch_bags,self.label_bags=self._generate_bags()
        self.positive_num=len(list(filter(lambda x: x==1, self.label_bags)))
        self.negative_num=len(list(filter(lambda x: x==0, self.label_bags)))
        
 
    def _generate_bags(self):
        
        train_patch_bags=[]
        train_label_bags=[]
        patches=[p for p in self.patch_paths]
        bag_names=list(self.labels.index)
        for i, b in enumerate(bag_names):
            image_patches=[p for p in patches if b in p]
            bag_size=self.n_size if len(image_patches)>self.n_size else len(image_patches)
            if bag_size==0:
                continue
            if bag_size<len(image_patches):
                num_bags=int(np.floor(len(image_patches)/bag_size))
            else:
                num_bags=1
            label_bag=[self.labels.loc[b][self.name]]
            label_bag=[self.key[l] for l in label_bag]
            for bi in range(num_bags):
                bag=random.sample(image_patches,bag_size)
                train_patch_bags.append(bag)
                train_label_bags.append(torch.tensor(label_bag))
        return train_patch_bags,train_label_bags
            
            
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
