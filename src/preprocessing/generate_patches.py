import os
import glob

import numpy as np
import cv2


out_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/breastcancer/data/breakhis/patches'
image_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/breastcancer/data/breakhis/images'

image_paths=glob.glob(os.path.join(image_path,'*'))

for i, path in enumerate(image_paths):
    class_paths=glob.glob(os.path.join(path,'*'))
    class_folder=os.path.basename(path)
    if not os.path.exists(os.path.join(out_path,class_folder)):
        os.makedirs(os.path.join(out_path,class_folder))
    for c in class_paths:
        print(c)
        image_folder=os.path.basename(c)[:-4]
        if not os.path.exists(os.path.join(out_path,class_folder,image_folder)):            
            os.makedirs(os.path.join(out_path,class_folder,image_folder))
        image=cv2.imread(c)
        for i in range(0,28*16,28):
            for j in range(0,28*25,28):
                name=image_folder+'_'+str(i)+'_'+str(j)+'.png'
                print(name)
                save_path=os.path.join(out_path,class_folder,image_folder,name)
                image[i:i+28,j:j+28,:]
                cv2.imwrite(save_path,image)
