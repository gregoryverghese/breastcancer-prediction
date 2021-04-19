import os 
import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class ICIARBags(Dataset):
    def __init__(self,patch_path,n=200,key={},name='labels'):
        self.n_size=n
        self.name=name
        self.key=key
        self.image_paths=glob.glob(os.path.join(patch_path,'*'))
        self.patch_bags,self.label_bags=self._generate_bags()
        self.class_nums=np.unique(np.array(self.label_bags),return_counts=True) 
 
    def _generate_bags(self):
        train_patch_bags=[]
        train_label_bags=[]
        for i, path in enumerate(self.image_paths):
            class_paths=glob.glob(os.path.join(path,'*'))
            for c in class_paths:
                image=cv2.imread(c)
                patches=[image[i:i+28,j:j+28,:] for i in
                        range(0,28*16,28) for j in range(0,28*25,28)]
                train_patch_bags.append(torch.tensor(patches))
                train_label_bags.append(i)

        return train_patch_bags, train_label_bags


    def __len__(self):
        return len(self.patch_bags)
    
    
    def __getitem__(self,index):
        bag=self.patch_bags[index]
        label=self.label_bags[index]
        
        return bag,label
