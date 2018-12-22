#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse


# In[ ]:



model_config = argparse.ArgumentParser()

# environment
model_config.add_argument('--gpu_id', type=eval, default=0)
model_config.add_argument('--root_path', type=str, default='.')

# model attribute
model_config.add_argument('--model_type', type=str, default='mono_depth', help='choose the kind of network: mono_depth, vgg16, resnet50')
model_config.add_argument('--height', type=eval, default=256, help='input height of the image')
model_config.add_argument('--width', type=eval, default=512, help='input width of the image')
model_config.add_argument('--channel', type=eval, default=3, help="input image channel number")
model_config.add_argument('--dataset', type=str, default='kitti', help='kitti, cityspace')
model_config.add_argument('--dataset_root', type=str, default='kitti_zip_light')
model_config.add_argument('--batch_size', type=eval, default=8)

# training
model_config.add_argument('--mode', type=str, default='train', help='train/ valid/ test')
model_config.add_argument('--epoch', type=eval, default=50, help='epoch number on training')
model_config.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate of the generator\'s optimizer')
model_config.add_argument('--resume', type=str, default='')

# testing
model_config.add_argument('--output_directory', type=str, default='evaluation')

# loss
model_config.add_argument('--alpha', type=float, default=0.85, help='weight between SSIM and L1 in image loss')
model_config.add_argument('--disp_gradient_loss_weight', type=float, default=0.1)
model_config.add_argument('--lr_loss_weight', type=float, default=1.0)

# evaluation
model_config.add_argument('--gt_directory', type=str, default='ground_truth')
model_config.add_argument('--min_depth', type=float, default=1e-3)
model_config.add_argument('--max_depth', type=float, default=80)
model_config.add_argument('--pp', action='store_true')

def get_config():
    config, unparsed = model_config.parse_known_args()
    return config, unparsed

