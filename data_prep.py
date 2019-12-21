# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 23:10:58 2019

@author: -
"""

import os
from glob import iglob
from os.path import join,basename
import shutil
import random

data_path = './data/' 
train_path = './train'
test_path = './test'

# create training set
print('_'*30)
print('Creating training set....')
print('_'*30)

for file in iglob(join(data_path,'*')):
    save_path = join(train_path,basename(file))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img in iglob(join(file,'*')):
         
        shutil.copy2(img,save_path)
        print(img) 


print('_'*30)
print('Creating test set....')
print('_'*30)

for file in iglob(join(train_path,'*')):
    save_path = join(test_path, basename(file))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    total_imgs = [x for x in iglob(join(file,'*'))]

    rand_amt = 0.15 * len(total_imgs)  # select 15% of data from each category as testing set
    test_imgs= []
    for i in range(int(rand_amt)):
        img = random.choice(total_imgs)
        if img not in test_imgs:
            shutil.move(img,save_path)
            test_imgs.append(img)
        print(img)