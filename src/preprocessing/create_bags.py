import os 
import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class ICIARBags(Dataset):
    def __init__(self,patch_path,n=200,key={},name='labels'):
        self.n_size=n
        self.name=name
        self.key=key
        self.image_paths=glob.glob(os.path.join(patch_path,'*'))
        self.data=[]
        for i, path in enumerate(self.image_paths):
            class_paths=glob.glob(os.path.join(path,'*'))
            for c in class_paths:
                self.data.append((c,i))
        #self.class_nums=np.unique(np.array(self.label_bags),return_counts=True) 


    def __getitem__(self,index):
        random.shuffle(self.data)
        path,label=self.data[index]
        #image=cv2.imread(path)
        image = Image.open(path)
        patches=[]
        #print(np.array(image).shape)
        #t = transforms.Compose([transforms.CenterCrop((448,700))])
        transformations = transforms.Compose([
            transforms.ToTensor()
        ])
        #image=np.array(t(image))
        image=np.array(image)
        for i in range(0,28*16,28):
            for j in range(0,28*25,28):
                patch=transformations(image[i:i+28,j:j+28,:])
                patches.append(patch)
        
        patches=tuple(patches)
        bag=torch.stack(patches, 0)
        #bag=patches.permute(0,3,1,2)
        return bag, label


    def __len__(self):
        return len(self.data)
    
