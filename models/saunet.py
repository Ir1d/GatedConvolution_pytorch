import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .spectral import SpectralNorm
from .networks import GatedConv2dWithActivation, GatedDeConv2dWithActivation, SNConvWithActivation, get_pad

class SelfAttention(nn.Module):
    "Self attention layer for nd."
    def __init__(self, n_channels:int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels//8)
        self.key   = conv1d(n_channels, n_channels//8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

class Interpolate(nn.Module):
    def __init__(self, factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.factor = factor
        # self.size = size
        # self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.factor)
        # x = self.interp(x, scale_factor=self.factor, align_corners=True)
        return x

class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, n_in_channel=5):
        super(InpaintSANet, self).__init__()
        cnum = 32
        self.coarse_net = nn.Sequential(
            #input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(3, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            #downsample to 64
            GatedConv2dWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # atrous convlution
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # upsample
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            #Self_Attn(2*cnum, 'relu'),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            #Self_Attn(cnum//2, 'relu'),
            GatedConv2dWithActivation(cnum//2, 1, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        """
        self.refine_conv_net = nn.Sequential(
            # input is 5*256*256
            GatedConv2dWithActivation(3, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample
            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample
            GatedConv2dWithActivation(2*cnum, 2*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )
        """
        self.refine_conv_net1 = nn.Sequential(
            # input is 5*256*256
            nn.Conv2d(1, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(cnum)
        )
        self.refine_conv_net2 = nn.Sequential(
            # downsample
            nn.Conv2d(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(2*cnum)
        )
        self.refine_conv_net3 = nn.Sequential(
            # downsample
            nn.Conv2d(2*cnum, 2*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(4*cnum),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(4*cnum),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(4*cnum),
            #Self_Attn(4*cnum, 'relu'),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            SelfAttention(4*cnum)
        )

        # self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=True)
        self.refine_attn = SelfAttention(4*cnum)
        # self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=False)
        """
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            #Self_Attn(cnum, 'relu'),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )
        """
        self.Up = Interpolate(2)
        self.refine_upsample_net1 = nn.Sequential(
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(2),
            SelfAttention(4*cnum),
            nn.Conv2d(4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.refine_upsample_net2 = nn.Sequential(
            # cnum channels from skip connection (map2)
            nn.Conv2d(2*cnum + 2 * cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(2),
            SelfAttention(2*cnum),
            nn.Conv2d(2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.refine_upsample_net3 = nn.Sequential(
            # cnum channels from skip connection (map1)
            nn.Conv2d(cnum + cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(cnum//2),
            #Self_Attn(cnum, 'relu'),
            nn.Conv2d(cnum//2, 3, 3, 1, padding=get_pad(256, 3, 1)),
        )
        self.ww = nn.Tanh()
        self.ff = nn.Conv2d(4, 3, 1)

    def forward(self, gray, masks, img_exs=None):
        # Coarse
        masked_imgs =  gray * (1 - masks) + masks
        if img_exs == None:
            input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        # else:
        #     input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        # input_imgs = torch.cat([masked_imgs], dim=1)
        #print(input_imgs.size(), gray.size(), masks.size())
        # print('input_size', input_imgs.size())
        x = self.coarse_net(input_imgs)
        # print(x.detach().min(), x.detach().max())
        # x = self.ww(x)
        # print(x.detach().min(), x.detach().max())
        x = torch.clamp(x, -1., 1.)
        coarse_x = x
        # Refine
        masked_imgs = gray * (1 - masks) + coarse_x * masks
        # masked_imgs = coarse_x
        # if img_exs is None:
        #     input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        # else:
            # input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
            
        input_imgs = torch.cat([masked_imgs], dim=1)
        # input_imgs = torch.cat([masked_imgs, gray], dim=1)
        map_1 = self.refine_conv_net1(input_imgs) # before downsample
        map_2 = self.refine_conv_net2(map_1) # downsample 1x
        
        x = self.refine_conv_net3(map_2)
        # x = self.refine_attn(x)
        # x, attention = self.refine_attn(x)
        #print(x.size(), attention.size())
        x = self.refine_upsample_net1(x)
        x = torch.cat([x, map_2], dim=1) # downsample 1x
        x = self.refine_upsample_net2(x)
        x = torch.cat([x, map_1], dim=1) # before downsample (original size)
        x = self.refine_upsample_net3(x)
        # x = self.ww(x)
        refined = torch.clamp(x, -1., 1.)
        stacked = torch.cat([coarse_x, refined], dim=1)
        x = self.ff(stacked)
        x = torch.clamp(x, -1., 1.)
        x = F.tanh(x)
        return coarse_x, refined, x#, attention
        # return coarse_x, x, attention

class InpaintSADirciminator(nn.Module):
    def __init__(self):
        super(InpaintSADirciminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(3, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
            SelfAttention(8*cnum, 'relu'),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        #x = self.linear(x)
        return x

