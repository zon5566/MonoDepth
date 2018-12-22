#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import re
import random
import zipfile
import glob
import time


# In[4]:


class Dataloader(Dataset):
    
    left_images = []
    right_images = []
    
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.left_images = []
        self.right_images = []
        self.archives = {}
        
        self.transform = transforms.Compose ([
                transforms.Resize(size=(params.height, params.width), interpolation=Image.BICUBIC),
                transforms.ToTensor()
            ]
        )
        
        # load image-path data which stores the absolute path where the image is
        nameLen = 66 # the length of image file name
        if self.mode == 'train':
            filepath_arr = np.genfromtxt(os.path.join('utils', 'kitti_train_files.txt'), dtype='U'+str(nameLen), delimiter=' ')
        elif self.mode == 'valid':
            filepath_arr = np.genfromtxt(os.path.join('utils', 'kitti_val_files.txt'), dtype='U'+str(nameLen), delimiter=' ')
        elif self.mode == 'test':
            filepath_arr = np.genfromtxt(os.path.join('utils', 'kitti_test_files.txt'), dtype='U'+str(nameLen), delimiter=' ')
        elif self.mode == 'evaluate':
            filepath_arr = np.genfromtxt(os.path.join('utils', 'kitti_stereo_2015_test_files.txt'), dtype='U30', delimiter=' ')
        
        if self.mode == 'evaluate':
            for line in filepath_arr:
                self.left_images.append(os.path.join('evaluation/ground_truth', line[0])[:-4] + '.png')
                self.right_images.append(os.path.join('evaluation/ground_truth', line[1])[:-4] + '.png')
        
        else:
            for line in filepath_arr:
                self.left_images.append(line[0])
                self.right_images.append(line[1])

            # load the zip file to the memory
            zippath = glob.glob('{}/{}/*_drive_*_sync.zip'.format(self.params.root_path, self.params.dataset_root))
            for path in zippath:
                archive = zipfile.ZipFile(path, 'r')
                pattern = re.compile('{}/{}/(.*).zip'.format(self.params.root_path, self.params.dataset_root))
                name = pattern.findall(path)[0]
                self.archives[name] = archive
        
    def __len__(self):
        return len(self.left_images)
    
    def __getitem__(self, index):
        
        if self.params.mode == 'evaluate':
            # It is loaded from disk, but we prefer to load from memory.
            left_img = Image.open(self.left_images[index])
            right_img = Image.open(self.right_images[index])
            
        else:
            left_img_path = self.left_images[index]
            right_img_path = self.right_images[index]

            folders = left_img_path.split('/')
            with self.archives[folders[1]].open(left_img_path) as file:
                left_img = Image.open(file)

            folders = right_img_path.split('/')
            with self.archives[folders[1]].open(right_img_path) as file:
                right_img = Image.open(file)
        
        arg = random.random() > 0.5
        if arg:
            left_img, right_img = self.augment_image_pair(left_img, right_img)
        
        # Resize the image, torchvision.transforms.functional.resize only accepts PIL format as input
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        return {'left_image': left_img.cuda(), 'right_image': right_img.cuda()}
        
    def augment_image_pair(self, left_image, right_image):
        
        # Change the order to fit the Pytorch format
        left_image = np.array(left_image)#.transpose((2,0,1))
        right_image = np.array(right_image)#.transpose((2,0,1))
        
        # randomly shift gamma
        gamma = random.uniform(0.8, 1.2)
        left_image_gamma = left_image**gamma
        right_image_gamma = right_image**gamma
        
        # randomly shift brightness
        brightness = random.uniform(0.5, 2.0)
        left_image_brightness = left_image_gamma * brightness
        right_image_brightness = right_image_gamma * brightness
        
        # randomly shift color
        colors = np.random.uniform(0.8, 1.2, 3)
        colors_map = np.tile(colors.reshape(1,1,self.params.channel), (left_image_brightness.shape[0], left_image_brightness.shape[1], 1))
        left_image_colors = left_image_brightness * colors_map
        right_image_colors = right_image_brightness * colors_map
        
        # saturate
        left_image_aug = np.clip(left_image_colors, 0, 255)
        right_image_aug = np.clip(right_image_colors, 0, 255)
        
        left_image_aug = Image.fromarray(np.uint8(left_image_aug))
        right_image_aug = Image.fromarray(np.uint8(right_image_aug))
        
        return left_image_aug, right_image_aug    

