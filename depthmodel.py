#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as transforms
import numpy as np
import model_monodepth
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mat
mat.use('Agg')


# In[2]:


class DepthModel(nn.Module):
    
    def __init__(self, params):
        super(DepthModel, self).__init__()
        
        self.params = params
        
        # The encoder-decoder model. It can be monodepth-model, ResNet-50
        if self.params.model_type.find('mono_depth') >= 0:
            from model_monodepth import Net
            self.G = Net(self.params.channel)
        elif self.params.model_type == 'resnet50':
            import model_resnet
            self.G = model_resnet.Resnet50_md(self.params.channel)
        
        else:
            print ('not available model type')
            return
        
        self.G.apply(self.weight_init)
    
    def set_cuda(self):
        self.G = self.G.cuda()
    
    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def scale_pyramid(self, img, num_scales):
        s = img.shape # expect BxCxHxW
        h, w = s[2], s[3]
        scaled_imgs = [img]
        
        for i in range(num_scales-1):
            ratio = 2**(i+1)
            nh = int(h // ratio)
            nw = int(w // ratio)
            img_up = functional.interpolate(img, size=(nh,nw), mode='nearest')
            scaled_imgs.append(img_up)
            
        return scaled_imgs
    
    def generate_image_left(self, img, disp):
        #return self.bilinear_sampler(img, -disp)
        return self.apply_disparity(img, -disp)
    
    def generate_image_right(self, img, disp):
        #return self.bilinear_sampler(img, disp)
        return self.apply_disparity(img, disp)
    
    def gradient_x(self, img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]
    
    def gradient_y(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]
    
    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]
        
        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]
        
        weights_x = [torch.exp(-torch.mean(g.abs(), dim=1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(g.abs(), dim=1, keepdim=True)) for g in image_gradients_y]
        
        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        
        return smoothness_x + smoothness_y
        
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        AvgPool2d = nn.AvgPool2d(3, stride=1)
        mu_x = AvgPool2d(x)
        mu_y = AvgPool2d(y)
        
        sigma_x = AvgPool2d(x.pow(2)) - mu_x.pow(2)
        sigma_y = AvgPool2d(y.pow(2)) - mu_y.pow(2)
        sigma_xy = AvgPool2d(x*y) - mu_x * mu_y
        
        SSIM_n = (2*mu_x*mu_y + C1) * (2*sigma_xy+C2)
        SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        
        return torch.clamp((1-SSIM)/2, 0, 1)
    
    def bilinear_sampler(self, images, x_offset, wrap_mode='border'):
        
        n_batch, n_channel, n_height, n_width = images.shape
        
        '''
        x = torch.linspace(0, 1, n_width).cuda()
        y = torch.linspace(0, 1, n_height).cuda()
        
        x = x.view(1, 1, -1).repeat(n_batch, n_height, 1)
        y = y.view(1, -1, 1).repeat(n_batch, 1, n_width)
        '''
        
        x = torch.linspace(0, 1, n_width).repeat(n_batch, n_height, 1).type_as(images)
        y = torch.linspace(0, 1, n_height).repeat(n_batch, n_width, 1).transpose(1, 2).type_as(images)
        
        offset = x_offset[:, 0, :, :]
        x = x + offset
        
        xy_grid = torch.stack((x,y), dim=3) # BxHxWx2
        xy_grid = xy_grid * 2 - 1 # map range(0,1) to range(-1, 1)
        images_generated = functional.grid_sample(images, xy_grid, padding_mode=wrap_mode)
        
        return images_generated
        
    def apply_disparity(self, input_images, x_offset, wrap_mode='border', tensor_type = 'torch.cuda.FloatTensor'):
        num_batch, num_channels, height, width = input_images.size()

        # Handle both texture border types
        edge_size = 0
        if wrap_mode == 'border':
            edge_size = 1
            # Pad last and second-to-last dimensions by 1 from both sides
            input_images = functional.pad(input_images, (1, 1, 1, 1))
        elif wrap_mode == 'edge':
            edge_size = 0
        else:
            return None

        # Put channels to slowest dimension and flatten batch with respect to others
        input_images = input_images.permute(1, 0, 2, 3).contiguous()
        im_flat = input_images.view(num_channels, -1)

        # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
        # meshgrid function)
        x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type).cuda()
        y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type).cuda()
        # Take padding into account
        x = x + edge_size
        y = y + edge_size
        # Flatten and repeat for each image in the batch
        x = x.view(-1).repeat(1, num_batch)
        y = y.view(-1).repeat(1, num_batch)

        # Now we want to sample pixels with indicies shifted by disparity in X direction
        # For that we convert disparity from % to pixels and add to X indicies
        x = x + x_offset.contiguous().view(-1) * width
        # Make sure we don't go outside of image
        x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
        # Round disparity to sample from integer-valued pixel grid
        y0 = torch.floor(y)
        # In X direction round both down and up to apply linear interpolation
        # between them later
        x0 = torch.floor(x)
        x1 = x0 + 1
        # After rounding up we might go outside the image boundaries again
        x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

        # Calculate indices to draw from flattened version of image batch
        dim2 = (width + 2 * edge_size)
        dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
        # Set offsets for each image in the batch
        base = dim1 * torch.arange(num_batch).type(tensor_type).cuda()
        base = base.view(-1, 1).repeat(1, height * width).view(-1)
        # One pixel shift in Y  direction equals dim2 shift in flattened array
        base_y0 = base + y0 * dim2
        # Add two versions of shifts in X direction separately
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        # Sample pixels from images
        pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
        pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

        # Apply linear interpolation to account for fractional offsets
        weight_l = x1 - x
        weight_r = x - x0
        output = weight_l * pix_l + weight_r * pix_r

        # Reshape back into image batch and permute back to (N,C,H,W) shape
        output = output.view(num_channels, num_batch, height, width).permute(1,0,2,3)

        return output
    
    def inference(self, test_input):
        self.test_disp = self.G.forward(test_input)
        return self.test_disp[0]
    
    def forward(self, left_image, right_image):

        self.left_pyramid = self.scale_pyramid(left_image, 4)
        if self.params.mode == 'train':
            self.right_pyramid = self.scale_pyramid(right_image, 4)
        
        # skip the stereo training part

        model_input = left_image
        
        self.disp_est = self.G.forward(model_input)
        self.disp_left_est = [torch.unsqueeze(d[:, 0, :, :], 1) for d in self.disp_est]
        self.disp_right_est = [torch.unsqueeze(d[:, 1, :, :], 1) for d in self.disp_est]
        
        if self.params.mode == 'test' or self.params.mode == 'evaluate': # Loss information is not needed in testing
            return
        
        # generate images
        self.left_est = [ self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
        self.right_est = [ self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]
        
        # LR consistency
        self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in range(4)]
        self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]
        
        # disparity smoothness
        self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)
        self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)
        
        self.build_loss()
        return self.total_loss.item()
        
    def build_loss(self):
        # L1
        self.l1_left = [torch.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
        self.l1_reconstruct_loss_left = [torch.mean(l) for l in self.l1_left]
        self.l1_right = [torch.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
        self.l1_reconstruct_loss_right = [torch.mean(l) for l in self.l1_right]
        
        # SSIM
        self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
        self.ssim_loss_left = [torch.mean(l) for l in self.ssim_left]
        self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
        self.ssim_loss_right = [torch.mean(l) for l in self.ssim_right]
        
        # weighted sum
        self.image_loss_left = [self.params.alpha * self.ssim_loss_left[i] + (1-self.params.alpha)*(self.l1_reconstruct_loss_left[i]) for i in range(4)]
        self.image_loss_right = [self.params.alpha * self.ssim_loss_right[i] + (1-self.params.alpha)*(self.l1_reconstruct_loss_right[i]) for i in range(4)]
        self.image_loss = sum(self.image_loss_left + self.image_loss_right)
        
        # disparity smoothness
        self.disp_left_loss = [torch.mean(torch.abs(self.disp_left_smoothness[i])) / 2**i for i in range(4)]
        self.disp_right_loss = [torch.mean(torch.abs(self.disp_right_smoothness[i])) / 2**i for i in range(4)]
        self.disp_gradient_loss = sum(self.disp_left_loss + self.disp_right_loss)
        
        # LR consistency
        self.lr_left_loss = [torch.mean(torch.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in range(4)]
        self.lr_right_loss = [torch.mean(torch.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
        self.lr_loss = sum(self.lr_left_loss + self.lr_right_loss)
        
        # total
        self.total_loss = (
                            self.image_loss + 
                            self.params.disp_gradient_loss_weight * self.disp_gradient_loss + 
                            self.params.lr_loss_weight * self.lr_loss )
        
    def display_image(self, path, epoch):
        '''
            left-image input, right-image input
            left-image est, right-image est
            left disp, right disp
        '''
        if not os.path.exists(path):
                os.makedirs(path)
        
        to_pil = transforms.Compose([transforms.ToPILImage()])
        
        for i in range(self.params.batch_size):
            plt.figure()
            
            plt.subplot(3,2,1)
            plt.imshow(np.asarray(to_pil(self.left_pyramid[0][i].cpu())))
            plt.title('left image input')
            plt.axis('off')
            
            plt.subplot(3,2,2)
            plt.imshow(np.asarray(to_pil(self.right_pyramid[0][i].cpu())))
            plt.title('right image input')
            plt.axis('off')
            
            plt.subplot(3,2,3)
            plt.imshow(np.asarray(to_pil(self.left_est[0][i].cpu())))
            plt.title('left image output')
            plt.axis('off')
            
            plt.subplot(3,2,4)
            plt.imshow(np.asarray(to_pil(self.right_est[0][i].cpu())))
            plt.title('right image output')
            plt.axis('off')
            
            plt.subplot(3,2,5)
            disp_image_left = self.disp_left_est[0][i].cpu().data.numpy()
            plt.imshow(disp_image_left.squeeze(), cmap=plt.get_cmap('plasma'))
            plt.title('left disparity estimation')
            plt.axis('off')
            
            plt.subplot(3,2,6)
            disp_image_right = self.disp_right_est[0][i].cpu().data.numpy()
            plt.imshow(disp_image_right.squeeze(), cmap=plt.get_cmap('plasma'))
            plt.title('right disparity estimation')
            plt.axis('off')
            
            plt.savefig('{}/img{:02d}-epoch{:02d}.jpg'.format(path, i+1, epoch), dpi=100)
            plt.close()    

