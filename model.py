from torch import nn,sigmoid, tanh
from torch.nn import functional as F
import numpy as np
import math

import utils


class ResidualBlock(nn.Module):
    def __init__(self, channels,  in_channels=None,dilation=None, kernel_size=3, dropout=0.25, norm="bn"):
        super(ResidualBlock, self).__init__()
        # if in_channels is None:
        #     in_channels=out_channels
        # self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        # # self.bn1 = nn.BatchNorm3d(out_channels)
        # self.bn1 = nn.InstanceNorm3d(out_channels)
        # self.prelu = nn.PReLU()
        # self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        # # self.bn2 = nn.BatchNorm3d(out_channels)
        # self.bn2 = nn.InstanceNorm3d(out_channels)
        # self.dropout = nn.Dropout(0.33)

        if dilation is None:
            padding=utils.calc_pad(kernel_size,1)
            
            self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=kernel_size,padding=padding)
            self.relu1 = nn.PReLU()
            # self.bn1 = nn.InstanceNorm3d(channels)
            # self.bn1 = nn.GroupNorm2d(channels,8)

            self.conv2 = nn.Conv3d(channels, channels, kernel_size=kernel_size,padding=padding)
            self.relu2 = nn.PReLU()
            # self.bn2 = nn.InstanceNorm3d(channels)
            self.dropout = nn.Dropout(dropout)
            # self.bn2 = nn.GroupNorm2d(channels,8)
        else:
            padding=utils.calc_pad(kernel_size,dilation)
            self.conv1 = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, dilation= dilation)
            # self.bn1 = nn.InstanceNorm3d(channels)
            # self.relu = nn.PReLU()
            self.relu1 = nn.PReLU()
            
            self.conv2 = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, dilation= dilation)
            self.relu2 = nn.PReLU()
            # self.bn2 = nn.InstanceNorm3d(channels)
            self.dropout = nn.Dropout(dropout)

        if norm == "bn":
            self.bn1 = nn.BatchNorm3d(channels)
            self.bn2 = nn.BatchNorm3d(channels)
        elif norm == "in":
            self.bn1 = nn.InstanceNorm3d(channels)
            self.bn2 = nn.InstanceNorm3d(channels)
            
    def forward(self, x):
#         print("Res input:"+str(x.size()))
        residual = self.conv1(x)
        residual = self.relu1(residual)

        residual = self.bn1(residual)
        # residual = self.prelu(residual)
        residual = self.conv2(residual)
        # residual = self.relu2(residual)
        residual = self.bn2(residual)
        residual = self.dropout(residual)
#         print("Res out:"+str(residual.size()))

        return x + residual
    
#Change    
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.upscale_factor=up_scale
        self.conv = nn.Conv3d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        #self.pixel_shuffle = pixel_shuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
#         print("upsample in:"+str(x.size()))
        x = self.conv(x)
        #x = self.pixel_shuffle(x)
        #FIXXX THIS
        input_size = list(x.size())
        dimensionality = len(input_size) - 2
#         print("IN size:"+str(input_size))

        input_size[1] //= (self.upscale_factor ** dimensionality)
#         print("IN 1 :"+str(input_size[1]))
        output_size = [dim * self.upscale_factor for dim in input_size[2:]]

        input_view = x.contiguous().view(
            input_size[0], input_size[1],
            *(([self.upscale_factor] * dimensionality) + input_size[2:])
        )

        indicies = list(range(2, 2 + 2 * dimensionality))
        indicies = indicies[1::2] + indicies[0::2]

        shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
        x = shuffle_out.view(input_size[0], input_size[1], *output_size)
        x = self.prelu(x)
#         print("upsample out:"+str(x.size()))
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d, dropout=0.25):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv3d(1, d, 3, 1, 1)
        self.relu1 = nn.PReLU()
        self.conv11 = nn.Conv3d(d, d, 3, 2, 1)
        self.relu11 = nn.PReLU()
        self.conv11_bn = nn.GroupNorm(16,d)
        self.dropout11 = nn.Dropout(dropout)

        self.conv2 = nn.Conv3d(d, d*2, 3, 1, 1)
        self.relu2 = nn.PReLU()
        self.conv2_bn = nn.GroupNorm(16,d*2)
        # self.conv2_bn = nn.InstanceNorm2d(d*2)
        self.conv22 = nn.Conv3d(d*2, d*2, 3, 2, 1)
        self.relu22 = nn.PReLU()
        # self.conv22_bn = nn.InstanceNorm2d(d*2)
        self.conv22_bn = nn.GroupNorm(16,d*2)
        self.dropout22 = nn.Dropout(dropout)

        self.conv3 = nn.Conv3d(d*2, d*4, 3, 1, 1)
        self.relu3 = nn.PReLU()
        self.conv3_bn = nn.GroupNorm(16,d*4)
        # self.conv3_bn = nn.InstanceNorm2d(d*4)
        self.conv33 = nn.Conv3d(d*4, d*4, 3, 2, 1)
        self.relu33 = nn.PReLU()
        self.conv33_bn = nn.GroupNorm(16,d*4)
        self.dropout33 = nn.Dropout(dropout)
        # self.conv33_bn = nn.InstanceNorm2d(d*4)
        self.conv4 = nn.Conv3d(d*4, d*8, 3, 1, 1)
        self.relu4 = nn.PReLU()
        self.conv4_bn = nn.GroupNorm(16,d*8)
        # self.conv4_bn = nn.InstanceNorm2d(d*8)
        self.conv44 = nn.Conv3d(d*8, d*8, 3, 2, 1)
        self.relu44 = nn.PReLU()
        self.conv44_bn = nn.GroupNorm(16,d*8)
        self.dropout44 = nn.Dropout(dropout)
        # self.conv44_bn = nn.InstanceNorm2d(d*8)
        #self.pool1=nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv3d(d*8, d*16, 3, 1, 1)
        self.relu5 = nn.PReLU()
        self.dropout5 = nn.Dropout(dropout)
        #self.conv5 = nn.Conv3d(d*4, d*8, 3, 1, 1)
    
    
        self.conv6 = nn.Conv3d(d*16, 1, 1)
        #self.conv6 = nn.Conv3d(d*8, 1, 1)   
    def weight_init(self, mean=0, std=0.01):
        for m in self._modules:
            utils.normal_init(self._modules[m], mean, std)       
            
        
    # forward method
    def forward(self, input):
#         print("Disc in:"+str(input.size()))

        x = self.relu1(self.conv1(input))
        x = self.relu11(self.conv11_bn(self.conv11(x)))
        x = self.dropout11(x)
        x = self.relu2(self.conv2_bn(self.conv2(x)))
        x = self.relu22(self.conv22_bn(self.conv22(x)))
        x = self.dropout22(x)
        x = self.relu3(self.conv3_bn(self.conv3(x)))
        x = self.relu33(self.conv33_bn(self.conv33(x)))
        x = self.dropout33(x)
        x = self.relu4(self.conv4_bn(self.conv4(x)))
        x = self.relu44(self.conv44_bn(self.conv44(x)))
        x = self.dropout44(x)
        x = self.relu5(self.conv5(x))
        x = self.dropout5(x)
        
        x = F.sigmoid(self.conv6(x))

        return x       

class Generator(nn.Module):
    # initializers
    def __init__(self,d, scale, no_of_dil_blocks, downsample, dropout=0.25, norm="bn"):
        
        super(Generator, self).__init__()
        
        self.downsample=downsample
        upsample_block_num = int(math.log(scale, 2))
        self.channels=d
        self.block1 = nn.Sequential(
            nn.Conv3d(1, self.channels, kernel_size=9, padding=4),
            nn.PReLU()
        )


        model_sequence=[]
        kernel_size=3
        self.no_of_dil_blocks=no_of_dil_blocks
        exp=-1
        for i in range(self.no_of_dil_blocks):
            # if i<=0:
            #     kernel_size=5
            exp=utils.update_exp(exp)
            model_sequence+=[ResidualBlock(self.channels,None,2**exp,kernel_size,dropout,norm)]
            # exp=utils.update_exp(exp)
            model_sequence+=[ResidualBlock(self.channels,None,2**exp,kernel_size,dropout,norm)]
            # exp=utils.update_exp(exp)
            model_sequence+=[ResidualBlock(self.channels,None,2**exp,kernel_size,dropout,norm)]
        self.model=nn.Sequential(*model_sequence)
        
        self.blockSLast = nn.Sequential(
            nn.Conv3d(d, d,  kernel_size=3,padding=1),
            nn.PReLU()
        )
        
        # FIXXX LATER - Think about using larger kernel_size
        # block_UpSample = [UpsampleBLock(d, 2) for _ in range(upsample_block_num)]
        # block_UpSample.append(nn.Conv3d(d // 2, d // 2, kernel_size=5, padding=2))
        # self.block_UpSample = nn.Sequential(*block_UpSample)
        
        # self.poolBlock1D = nn.MaxPool3d(3,stride=2,padding=1)
        if downsample:
            self.block_pad = nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0.0)
            self.blockLast = nn.Sequential(
                # nn.Conv3d(d // 2, 1,  kernel_size=3,padding=1),
                nn.Conv3d(d, 1,  kernel_size=3, padding=1),
                # nn.BatchNorm3d(1),
                nn.PReLU()
            )
        else:
            self.blockLast = nn.Sequential(
                # nn.Conv3d(d // 2, 1,  kernel_size=3, stride=2, padding=1),
                nn.Conv3d(d, 1,  kernel_size=3, padding=1),
                # nn.BatchNorm3d(1),
                nn.PReLU()
            )            
        
    def weight_init(self, mean=0, std=0.01):
        for m in self._modules:
            utils.normal_init(self._modules[m], mean, std)

        # forward method
    def forward(self, input):
        # print("G imnput:"+str(input.size()))
        block1 = self.block1(input)
        x=self.model(block1)
        x=self.blockSLast(x)
        # blockSEnd = self.block_UpSample(block1 + x)
        blockSEnd = (block1 + x)
        # print("BlockSend:"+str(blockSEnd.size()))
        blockPad=blockSEnd
        if self.downsample:
            blockPad=self.block_pad(blockSEnd)
        # print("blockpad"+str(blockPad.size()))
        blockEnd = self.blockLast(blockPad)
        # x = blockEnd
        x = (tanh(blockEnd) + 1) / 2

        # print("After block last:"+str(x.size()))
        return x