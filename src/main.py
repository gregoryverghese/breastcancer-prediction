import os
import glob
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import Trainer
from preprocessing.create_bags import ICIARBags
from models.gated_attention import AttentionGated
from models.attention import Attention

print('initiating...')
print(torch.cuda.is_available())
#image_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/breastcancer/data/breakhis/images'
image_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/breastcancer/data/breakhis/images'
#need to think about how to organise patches to call them in
#here we generate image level details 
#label_df=pd.read_csv(label_path)
#label_df['patch_names']=label_df['patch_names'].map(lambda x: x.split('._')[0])
#del label_df['Unnamed: 0']
#label_df=label_df.drop_duplicates()
#label_df.to_csv('/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/hormad1/data/patches/image_labels.csv')

print('getting patch dataset...')
#patch_dataset=ICIARBags(image_path,n=100)
#print('numbags: {}'.format(len(patch_dataset.patch_bags)))
patch_dataset=ICIARBags(image_path)

print('getting train loader...')
train_loader = DataLoader(patch_dataset,batch_size=1,
                          shuffle=True,num_workers=16)

print('number bags: {}'.format(len(patch_dataset)))
#valid_loader = DataLoader(patch_dataset,batch_size=1,shuffle=True)

#print('number of classes: {}'.format(patch_dataset.class_nums))
num=0
for b in train_loader:
    #num+=len(b[0])
    pass
print(b[0].size())
#print('negative:{},positive:{}'.format(patch_dataset.negative_num,patch_dataset.positive_num))


valid_loader=None

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=AttentionGated()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainer=Trainer(model,device,optimizer,train_loader, valid_loader,100)
trainer.run()

