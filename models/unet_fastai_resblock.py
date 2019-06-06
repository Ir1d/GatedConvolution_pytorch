import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .spectral import SpectralNorm
from .networks import GatedConv2dWithActivation, GatedDeConv2dWithActivation, SNConvWithActivation, get_pad
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out

class SAGenerator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(SAGenerator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2

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

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        
    def forward(self, x):
        return x * F.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    # originally 9 resblocks
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, 64, 3),
                 Swish()]
                #  nn.LeakyReLU(0.2, inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            self.add_module('down' + str(_+1), 
                nn.Sequential(nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    Swish())
            )
                    # nn.LeakyReLU(0.2, inplace=True))
            # model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    #   nn.LeakyReLU(0.2, inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        model1 = []
        for _ in range(n_residual_blocks):
            model1 += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            self.add_module('up' + str(_+1), 
                nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                      Swish())
            )
                    #   nn.LeakyReLU(0.2, inplace=True))
            # model += [nn.Upsample(scale_factor=2, mode='nearest'),
            #           nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            #           nn.LeakyReLU(0.2, inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model2 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, output_nc, 3),
                  nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
    def forward(self, input):
        x = self.model(input)
        map1 = self.down1(x)
        map2 = self.down2(map1)

        x = self.model1(map2)

        x = self.up1(x + map2)
        x = self.up2(x + map1)

        x = self.model2(x)
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
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.refine_conv_net2 = nn.Sequential(
            # downsample
            nn.Conv2d(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.refine_conv_net3 = nn.Sequential(
            # downsample
            nn.Conv2d(2*cnum, 2*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.resBlock1 = nn.Sequential(
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.resBlock2 = nn.Sequential(
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            #Self_Attn(4*cnum, 'relu'),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )

        # self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=True)
        self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=False)
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
            nn.Conv2d(4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.refine_upsample_net2 = nn.Sequential(
            # cnum channels from skip connection (map2)
            nn.Conv2d(2*cnum + 2 * cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(2),
            nn.Conv2d(2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.refine_upsample_net3 = nn.Sequential(
            # cnum channels from skip connection (map1)
            nn.Conv2d(cnum + cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            #Self_Attn(cnum, 'relu'),
            nn.Conv2d(cnum//2, 3, 3, 1, padding=get_pad(256, 3, 1)),
        )
        self.ww = nn.Tanh()
        self.ff = nn.Conv2d(4, 3, 1)
        self.unet = Generator(1, 3)

    def forward(self, imgs, masks, gray, img_exs=None):
        # print(len(_))
        # for x in ww:
        #     print(x.shape)
        # imgs, masks, gray = ww
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
        """
        map_1 = self.refine_conv_net1(input_imgs) # before downsample
        map_2 = self.refine_conv_net2(map_1) # downsample 1x
        
        x = self.refine_conv_net3(map_2)
        y = self.resBlock1(x)
        x = x + y
        y = self.resBlock2(x)
        x = x + y
        # change the bottleneck layers to resBlock

        x = self.refine_attn(x)
        # x, attention = self.refine_attn(x)
        #print(x.size(), attention.size())
        x = self.refine_upsample_net1(x)
        x = torch.cat([x, map_2], dim=1) # downsample 1x
        x = self.refine_upsample_net2(x)
        x = torch.cat([x, map_1], dim=1) # before downsample (original size)
        x = self.refine_upsample_net3(x)
        """
        x = self.unet(input_imgs)
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
            Self_Attn(8*cnum, 'relu'),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        #x = self.linear(x)
        return x


class SADiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2
