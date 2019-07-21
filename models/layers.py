import torchvision
import random
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np




class ResidualBlock(nn.Module):
    def __init__(self, in_features=None,out_features=None,k=None):
        super(ResidualBlock, self).__init__()
        if k is None:
            k = 3
        padd = int(np.floor(k/2))
        conv_block = [  nn.ReflectionPad2d(padd),
                        nn.Conv2d(in_features, in_features, k),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(padd),
                        nn.Conv2d(in_features, out_features, k),
                        nn.InstanceNorm2d(out_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3),
                        nn.InstanceNorm2d(out_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))
        
    def forward(self, image, mask):
#         pdb.set_trace()
        noise = torch.randn(1, 1, image.shape[2], image.shape[3]).cuda()
        mask = mask[:,:1,:,:].repeat(1,image.shape[1],1,1)
        return image + self.weight * noise * mask
    
class model_ds(nn.Module):
    def __init__(self, in_features,out_features):
        super(model_ds, self).__init__()

        conv_block = [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(inplace=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)  


class model_up(nn.Module):
    def __init__(self, in_features,out_features):
        super(model_up, self).__init__()

        conv_block = [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)      

def swish(x):
    return x * F.sigmoid(x)

def get_mean_var(c):
    n_batch, n_ch, h, w = c.size()

    c_view = c.view(n_batch, n_ch, h * w)
    c_mean = c_view.mean(2)

    c_mean = c_mean.view(n_batch, n_ch, 1, 1).expand_as(c)
    c_var = c_view.var(2)
    c_var = c_var.view(n_batch, n_ch, 1, 1).expand_as(c)
    # c_var = c_var * (h * w - 1) / float(h * w)  # unbiased variance

    return c_mean, c_var


    
    
class transform_layer(nn.Module):
    
    def __init__(self,in_features,out_features):
        super(transform_layer, self).__init__()
        self.channels = in_features
        

        self.convblock = ConvBlock(in_features+in_features,out_features)
        self.up_conv = nn.Conv2d(in_features*2,in_features,3,1, 1)
        self.down_conv = nn.Sequential(
            nn.Conv2d(64,in_features//4,3,1, 1),
            nn.ReLU(),
            nn.Conv2d(in_features//4,in_features//2,1,1),
            nn.ReLU(),
            nn.Conv2d(in_features//2,in_features,1,1),
            nn.ReLU()
        )  
        self.noise = NoiseInjection(in_features)
        
        
        
        self.convblock_ = ConvBlock(in_features+64,out_features)

        self.vgg_block = nn.Sequential(
            nn.Conv2d(4,16,3,1, 1),
            nn.ReLU(),
            nn.Conv2d(16,32,1,1),
            nn.ReLU(),
            nn.Conv2d(32,64,1,1),
            nn.ReLU()
        ) 
       
    def forward(self,x,mask=None,style=None,mode='D'):
#         pdb.set_trace()
        if mode=='C':
            style = F.upsample(style, size=(x.shape[2],x.shape[2]), mode='bilinear')

            style = self.vgg_block(style)
            concat = torch.cat([x,style],1)

            out = (self.convblock_(concat))
            return out, style
        else:
            mask = F.upsample(mask, size=(x.shape[2],x.shape[2]), mode='bilinear')
            x = self.noise(x,mask)
#             style = F.upsample(style, size=(x.shape[2],x.shape[2]), mode='bilinear')

            style = self.down_conv(style)
            concat = torch.cat([x,style],1)

            out = (self.convblock(concat) + style)
            return out
        
        
class transform_up_layer(nn.Module):
    
    def __init__(self,in_features,out_features,diff=False):
        super(transform_up_layer, self).__init__()
        self.channels = in_features
        
        if diff ==True:
            self.convblock = ConvBlock(in_features*2+in_features,out_features)
        else:
            self.convblock = ConvBlock(in_features*2,out_features)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_features*2,in_features,3,1, 1),
            nn.ReLU()
        )
        
    def forward(self,x,y,mode="down"):

        y = self.up_conv(y)
        concat = torch.cat([x,y],1)
        
        out = self.convblock(concat)
        
#         out = self.adain(out,style)
        
        return out
