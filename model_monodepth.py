#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as functional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[ ]:


class Net(nn.Module):

        def __init__(self, in_chn):
            super(Net, self).__init__()
            self.in_chn = in_chn
            
            # ======== MonoDepth ========#
            # Encoder
            self.downconv_1 = self.conv_down_block(self.in_chn, 32, 7)
            self.downconv_2 = self.conv_down_block(32, 64, 5)
            self.downconv_3 = self.conv_down_block(64, 128, 3)
            self.downconv_4 = self.conv_down_block(128, 256, 3)
            self.downconv_5 = self.conv_down_block(256, 512, 3)
            self.downconv_6 = self.conv_down_block(512, 512, 3)
            self.downconv_7 = self.conv_down_block(512, 512, 3)
            
            # Decoder
            self.upconv_7 = self.conv_up_block(512, 512, 3, 2)
            self.upconv_6 = self.conv_up_block(512, 512, 3, 2)
            self.upconv_5 = self.conv_up_block(512, 256, 3, 2)
            self.upconv_4 = self.conv_up_block(256, 128, 3, 2)
            self.upconv_3 = self.conv_up_block(128, 64, 3, 2)
            self.upconv_2 = self.conv_up_block(64, 32, 3, 2)
            self.upconv_1 = self.conv_up_block(32, 16, 3, 2)
            
            self.iconv_7 = self.iconv(1024, 512, 3)
            self.iconv_6 = self.iconv(1024, 512, 3)
            self.iconv_5 = self.iconv(512, 256, 3)
            self.iconv_4 = self.iconv(256, 128, 3)
            self.iconv_3 = self.iconv(130, 64, 3)
            self.iconv_2 = self.iconv(66, 32, 3)
            self.iconv_1 = self.iconv(18, 16, 3)
            
            self.disp4 = self.get_disp(128, 2)
            self.disp3 = self.get_disp(64, 2)
            self.disp2 = self.get_disp(32, 2)
            self.disp1 = self.get_disp(16, 2)
            
            '''
            # ======== ResNet50 ======== #
            # encoder
            self.conv_pad = Conv_pad(in_chn, 64, 7, 2)
            self.maxpool = nn.MaxPool2d(3)
            self.resblock1 = self.resblock(64, 3)
            self.resblock2 = self.resblock(128, 4)
            self.resblock3 = self.resblock(256, 6)
            self.resblock4 = self.resblock(512, 3)
            '''
        '''
        def resblock(self, in_chn, n_blocks):
            layers = []
            for _ in range(n_blocks):
                layers.append(ResConv(in_chn, in_chn, 1))
            layers.append(ResConv(in_chn, in_chn, 2))
            
            return nn.Sequential(*layers)
        '''
        def conv_down_block(self, in_chn, out_chn, kernel):
            return nn.Sequential (
                nn.ReplicationPad2d(kernel//2),
                nn.Conv2d(in_chn, out_chn, kernel_size=kernel, stride=1, padding=0),
                #nn.BatchNorm2d(out_chn),
                nn.ELU(),
                nn.ReplicationPad2d(kernel//2),
                nn.Conv2d(out_chn, out_chn, kernel_size=kernel, stride=2, padding=0),
                #nn.BatchNorm2d(out_chn),
                nn.ELU(),
                nn.Dropout(0.4)
            )
        
        def conv_up_block(self, in_chn, out_chn, kernel, scale):
            return nn.Sequential (
                nn.Upsample(scale_factor=scale, mode='nearest'),
                nn.ReplicationPad2d(kernel//2),
                nn.Conv2d(in_chn, out_chn, kernel_size=kernel, stride=1, padding=0),
                #nn.BatchNorm2d(out_chn),
                nn.ELU()
            )
        '''
        def conv_up_block_resnet(self, in_chn, out_chn, kernel_size, scale):
            return nn.Sequential (
                nn.Upsample(scale_factor=scale, mode='nearest'),
                Conv_pad(in_chn, out_chn, kernel_size, 1)
            )
        '''
        def iconv(self, in_chn, out_chn, kernel):
            return nn.Sequential (
                nn.ReplicationPad2d(kernel//2),
                nn.Conv2d(in_chn, out_chn, kernel_size=kernel, stride=1, padding=0),
                #nn.BatchNorm2d(out_chn),
                nn.ELU()
            )
        
        def get_disp(self, in_chn, out_chn, kernel=3):
            return nn.Sequential (
                nn.ReplicationPad2d(1),
                nn.Conv2d(in_chn, out_chn, kernel_size=kernel, stride=1, padding=0),
                #nn.BatchNorm2d(out_chn),
                nn.Sigmoid()
            )
        
        def build_monodepth(self, x):
            
            # Encoder
            conv1 = self.downconv_1(x)          #  32x128x256
            conv2 = self.downconv_2(conv1)  #  64x  64x128
            conv3 = self.downconv_3(conv2) # 128x  32x 64
            conv4 = self.downconv_4(conv3) # 256x  16x 32
            conv5 = self.downconv_5(conv4) # 512x   8x  16
            conv6 = self.downconv_6(conv5) # 512x   4x   8
            conv7 = self.downconv_7(conv6) # 512x    2x  4

            
            # Decoder
            upconv7 = self.upconv_7(conv7) # 512 x 4 x 8
            concat_7 = torch.cat((upconv7, conv6), 1)
            iconv7 = self.iconv_7(concat_7) # 512 x 4 x 8
          
            upconv6 = self.upconv_6(iconv7) # 512 x 8 x 16
            concat_6 = torch.cat((upconv6, conv5), 1)
            iconv6 = self.iconv_6(concat_6) # 512 x 8 x 16
            
            upconv5 = self.upconv_5(iconv6) # 256 x 16 x 32
            concat_5 = torch.cat((upconv5, conv4), 1)
            iconv5 = self.iconv_5(concat_5) # 256 x 16 x 32
            
            upconv4 = self.upconv_4(iconv5) # 128 x 32 x 64
            concat_4 = torch.cat((upconv4, conv3), 1)
            iconv4 = self.iconv_4(concat_4) # 128 x 32 x 64
            
            disp4 = 0.3*self.disp4(iconv4)
            up_disp4 = functional.interpolate(disp4, scale_factor=2)
            
            upconv3 = self.upconv_3(iconv4) # 64 x 64 x 128
            concat_3 = torch.cat((upconv3, conv2, up_disp4), 1)
            iconv3 = self.iconv_3(concat_3) # 64 x 64 x 128

            disp3 = 0.3*self.disp3(iconv3)
            up_disp3 = functional.interpolate(disp3, scale_factor=2)
            
            upconv2 = self.upconv_2(iconv3) # 32 x 128 x 256
            concat_2 = torch.cat((upconv2, conv1, up_disp3), 1)
            iconv2 = self.iconv_2(concat_2) # 32 x 128 x 256
            
            disp2 = 0.3*self.disp2(iconv2)
            up_disp2 = functional.interpolate(disp2, scale_factor=2)
            
            upconv1 = self.upconv_1(iconv2) # 16 x 256 x 512
            concat_1 = torch.cat((upconv1, up_disp2), 1)
            iconv1 = self.iconv_1(concat_1) # 16 x 256 x 512
            disp1 = 0.3*self.disp1(iconv1)
            
            return [disp1, disp2, disp3, disp4]
        
        '''
        def build_resnet50(self, x):
            
            # encoder
            p = kernel // 2
            x_pad = functional.pad(x, (p, p, p, p, 0, 0, 0, 0), 'constant')
            
            conv1 = self.conv_pad(x_pad)
            conv1_pad = functioanl.pad(conv1, (1,1,1,1,0,0,0,0), 'constant')
            pool1 = self.maxpool(conv1_pad)
            
            conv2 = self.resblock1(pool1)
            conv3 = self.resblock2(conv2)
            conv4 = self.resblock3(conv3)
            conv5 = self.resblock4(conv4)
            
            # skip layer
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
            
            # decoder
        '''
        def forward(self, x):
            out = self.build_monodepth(x)
            return out


# In[ ]:


'''
class Conv_pad(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride):
        super(Conv_pad, self).__init__()
        
        self.p = kernel_size // 2
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride)
        self.elu = nn.ELU()

    def forward(self, x):    
        x_pad = functional.pad(x, (self.p, self.p, self.p, self.p, 0, 0, 0, 0), 'constant')
        conv1 = self.conv(x_pad)
        out = self.elu(conv1)

        return out
    
class ResConv(nn.Module):
    def __init__(self, in_chn, out_chn, stride):
        super(ResConv, self).__init__()
        
        self.out_chn = out_chn
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=stride)
        self.conv3 = nn.Conv2d(out_chn, out_chn, kernel_size=1, stride=1)
        self.shortcut = nn.Conv2d(in_chn, 4*out_chn, kernel_size=1, stride=stride)
        
        self.elu = nn.ELU()
    
    def forward(self, x):
        do_proj = x.shape[1] != self.out_chn or self.stride == 2

        conv1 = self.conv1(x)
        conv1 = self.elu(conv1)
        
        conv2 = self.conv2(conv1)
        conv2 = self.elu(conv2)
        
        conv3 = self.conv3(conv2)
        
        if do_proj:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out = self.elu(conv3 + shortcut)
        
        return out
'''

