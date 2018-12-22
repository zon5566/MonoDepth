#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
import time
import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
import config
from depthmodel import DepthModel
import utils
from glob import glob
from dataloader import Dataloader as DL
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mat
mat.use('Agg')


# In[ ]:


def post_process_disparity(disp, disp_flip):
    h, w = disp.shape
    #l_disp = disp[0,:,:]
    #r_disp = np.fliplr(disp[1,:,:])
    l_disp = disp
    r_disp = disp_flip
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

if __name__ == '__main__':

    # network model and parameters
    params, _ = config.get_config()
    model = DepthModel(params)
    
    if torch.cuda.is_available():
        # GPU setting
        print("GPU Acceleration Available")
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        model.set_cuda()
        
    if params.mode == 'train':
        # loading training data and validation data
        data_loader_train= DL(params, 'train')
        train_data = DataLoader(data_loader_train, batch_size=params.batch_size, shuffle=True)  
        data_loader_valid = DL(params, 'valid')
        valid_data = DataLoader(data_loader_valid, batch_size=params.batch_size, shuffle=False)
        
        # optimizer
        opt = torch.optim.Adam(model.G.parameters(), lr=params.learning_rate)
        
        # tensorboard visualization
        writer = SummaryWriter('runs/{}'.format(params.model_type), comment='epoch{}_lr{}'.format(params.epoch, params.learning_rate))
        
        # resume training
        if params.resume:
            checkpoint = torch.load(os.path.join('checkpoints', params.model_type, params.resume), map_location=lambda storage, loc: storage)
            print ('=> loading checkpoint {}/{}, last training loss = {:.5f}'.format(params.model_type, params.resume, checkpoint['loss']))
            start_epoch = checkpoint['epoch']
            model.G.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler = MultiStepLR(opt, milestones=[int(params.epoch*(1/2)), int(params.epoch*(4/5))], gamma=0.5, last_epoch=start_epoch)
        else:
            start_epoch = 0
            scheduler = MultiStepLR(opt, milestones=[int(params.epoch*(1/2)), int(params.epoch*(4/5))], gamma=0.5, last_epoch=-1)

        # start training
        for epoch in range(start_epoch+1, params.epoch+1):
            
            epoch_start_time = time.time()
            scheduler.step()
            print ('{} epoch starts, learning rate = {}'.format(epoch, utils.get_lr(opt)))
            
            for i, data in enumerate(train_data):
                iter_start_time = time.time()
                loss = model.forward(data['left_image'], data['right_image'])
                opt.zero_grad()
                model.total_loss.backward()
                opt.step()    

                if i % 200 == 0:
                    iter_duration = time.time() - iter_start_time
                    print ('epoch {:2d} | iteration {:4d} | loss = {:.3f} (img={:.3f}, disp={:.5f}, consist={:3f}) | {:.2f} sec'                                 .format(epoch, i, loss, model.image_loss.item(), model.disp_gradient_loss.item(), model.lr_loss.item(), iter_duration))
           
            # save checkpoints at checkpoints/<model name>/cp_epoch<n>_lr<n>.pth
            if not os.path.exists('checkpoints/{}'.format(params.model_type)):
                os.makedirs('checkpoints/{}'.format(params.model_type))
            time.sleep(8)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.G.cpu().state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }, 'checkpoints/{}/cp_epoch{:02d}.pth'.format(params.model_type, epoch))
            model.G = model.G.cuda()
            
            # validation and display the images
            for d in valid_data:
                loss_v = model.forward(d['left_image'], d['right_image'])
                model.display_image('img/{}'.format(params.model_type), epoch)
                break
            
            # plot the loss chart in tensorboard
            writer.add_scalars('train_loss', {
                            'image loss': model.image_loss.item(), 
                            'Disparity smoothness': params.disp_gradient_loss_weight * model.disp_gradient_loss.item(),
                            'Consistency': params.lr_loss_weight * model.lr_loss.item()}, epoch)
            writer.add_scalars('train_valid_loss', {
                            'train loss': loss,
                            'valid loss': loss_v}, epoch)
            
            epoch_duration = time.time() - epoch_start_time
            print ('epoch {} spending time = {:.2f} sec\n'.format(epoch, epoch_duration))
            
        writer.close()
        
    elif params.mode == 'test' or params.mode == 'evaluate':
        
        # load testing dataset
        data_loader_test = DL(params, params.mode)
        test_data = DataLoader(data_loader_test, batch_size=params.batch_size, shuffle=False)
        
        # load trained model. Here we find the last epoch's checkpoint and load it
        cp_list = glob('checkpoints/{}/cp_epoch*.pth'.format(params.model_type))
        cp_filename = max(cp_list, key=lambda f: int(re.findall('\d+', f)[0]))
        print ('=> loading model {}'.format(cp_filename))
        cp = torch.load(cp_filename)
        model.G.load_state_dict(cp['model_state_dict'])
        
        disparities = np.zeros((len(test_data)*params.batch_size, params.height, params.width), dtype=np.float32)
        disparities_pp = np.zeros((len(test_data)*params.batch_size, params.height, params.width), dtype=np.float32)
        
        print ('Start testing {} samples...'.format(len(test_data)*params.batch_size))
        
        with torch.no_grad():
            for i, data in enumerate(test_data):
                model.forward(data['left_image'], data['right_image'])
                disp = model.disp_est[0]
                model.forward(torch.flip(data['left_image'], [3]), torch.flip(data['right_image'], [3]))
                disp_flip = model.disp_est[0]
                
                for j in range(params.batch_size):
                    
                    # disparity map
                    disp_np = disp[j,0,:,:].squeeze().cpu().numpy()
                    disparities[i*params.batch_size+j] = disp_np
                    '''
                    plt.figure()
                    plt.imshow(disp_np, cmap=plt.get_cmap('plasma'))
                    plt.axis('off')
                    plt.savefig('evaluation/img/{}/{:03d}.png'.format(params.model_type, i*params.batch_size + j))
                    plt.close()
                    '''
                    
                    # disparity map with post-processing
                    #disp_np_pp = disp[j].squeeze().cpu().data.numpy()
                    disp_np_pp = disp[j,0,:,:].squeeze().cpu().data.numpy()
                    disp_flip_np_pp = disp_flip[j,0,:,:].squeeze().cpu().data.numpy()
                    disparities_pp[i*params.batch_size+j] = post_process_disparity(disp_np_pp, disp_flip_np_pp[:,::-1])
                    '''
                    plt.figure()
                    plt.imshow(disp_np_pp, cmap=plt.get_cmap('plasma'))
                    plt.axis('off')
                    plt.savefig('evaluation/img/{}/{:03d}_pp.png'.format(params.model_type, i*params.batch_size + j))
                    plt.close()
                    '''
                    
        if not os.path.exists(params.output_directory):
            os.makedirs(params.output_directory)
        np.save('{}/npy/disparities_{}.npy'.format(params.output_directory, params.model_type), disparities)
        np.save('{}/npy/disparities_{}_pp.npy'.format(params.output_directory, params.model_type), disparities_pp)
        
        print ('Finished testing. Saved the output file at {}/npy/disparities_{}.npy'.format(params.output_directory, params.model_type))


# In[ ]:




