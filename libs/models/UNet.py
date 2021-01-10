import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import sync_batchnorm as syncBN

class GradBlock(nn.Module):
    def __init__(self,dim,channels):
        super(GradBlock, self).__init__()
    def forward(self, x):
        dx = x[:,:,2:,:,:] - x[:,:,:-2,:,:]
        dy = x[:,:,:,2:,:] - x[:,:,:,:-2,:]
        dz = x[:,:,:,:,2:] - x[:,:,:,:,:-2]
        out = torch.zeros(x.shape)
        out[:,:,1:-1,:,:] = dx**2
        out[:,:,:,1:-1,:] += dy**2
        out[:,:,:,:,1:-1] += dz**2
        return out

class BasicBlock(nn.Module):
    def __init__(self,block_depth,in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=True):
        super(BasicBlock, self).__init__()
        global conv_fn,bn_fn
        self.layer = []
        self.depth = block_depth
        self.layer.append(nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2)
        ))
        for _ in range(block_depth-1):
            self.layer.append(
                nn.Sequential(
                    conv_fn(out_channels, out_channels, 3, stride=1, padding=padding),
                    bn_fn(out_channels),
                    nn.LeakyReLU(0.2)
                )
            )
        self.net = nn.Sequential(*(self.layer))
    def forward(self, x):
        out = self.net(x)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self,block_depth,in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=True):
        super(ResidualBlock, self).__init__()
        global conv_fn,bn_fn
        self.layer = []
        self.depth = block_depth
        self.layer.append(nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2)
        )
        )
        for _ in range(block_depth-2):
            self.layer.append(
                nn.Sequential(
                    conv_fn(out_channels, out_channels, 3, stride=1, padding=padding),
                    bn_fn(out_channels),
                    nn.LeakyReLU(0.2)
                )
            )
        self.layer.append(
            nn.Sequential(
                conv_fn(out_channels, out_channels, 3, stride=1, padding=padding),
                bn_fn(out_channels)
            )
        )
        self.net = nn.Sequential(*(self.layer))
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    conv_fn(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    bn_fn(out_channels)
            )
    def forward(self, x):
        out = self.net(x)
        out = self.shortcut(x)+out
        out = F.leaky_relu(out,0.2)
        return out

class Encoder(nn.Module):
    def __init__(self, input_channel,channel_expansion=16,blocks = ResidualBlock, block_depth = 1, block_num=4):
        super(Encoder,self).__init__()
        self.image_channel = input_channel
        self.block_num = block_num
        self.enc = nn.ModuleList()
        for i in range(block_num):
            input_channel = self.image_channel if i == 0 else channel_expansion*(2**(i-1))
            output_channel = channel_expansion*(2**(i))
            stride = 1 if i==0 else 2
            self.enc.append(blocks(block_depth,  int(input_channel), int(output_channel), kernel_size = 3, stride=stride))
    def forward(self,x):
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        return x_enc

class Decoder(nn.Module):
    def __init__(self, channel_expansion=16,blocks = ResidualBlock,block_depth = 1, block_num=4):
        super(Decoder,self).__init__()
        self.block_num = block_num
        self.dec = nn.ModuleList()
        for i in range(block_num):
            input_channel = channel_expansion*(2**(block_num-i-1)) if i == 0 else channel_expansion*(2**(block_num-i))
            output_channel = channel_expansion*(2**(block_num-i-2))
            self.dec.append(blocks(block_depth, int(input_channel), int(output_channel)))
        self.output_channel = int(output_channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,x_enc):
        y = x_enc[-1]
        for i in range(self.block_num-1):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        y = self.dec[-1](y)
        return y


class UNet(nn.Module):
    def __init__(self, dim, input_channel, n_classes,block_depth=2,block_num=4, bn=None, full_size=True, bn_type = 'batch'):
        super(UNet, self).__init__()
        self.dim = dim
        global conv_fn, bn_fn
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        if bn_type == 'batch':
            bn_fn = getattr(syncBN, "SynchronizedBatchNorm{0}d".format(dim))
        else:
            bn_fn = getattr(nn,'InstanceNorm{0}d'.format(dim))
        self.encoder = Encoder(input_channel,block_depth=block_depth,block_num=block_num)
        self.decoder = Decoder(block_depth=block_depth,block_num=block_num)
        # One conv to get the segmentation
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.out_conv = conv_fn(self.decoder.output_channel, n_classes, kernel_size=3, padding=1)
        
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Get encoder activations
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        y = self.out_conv(x_dec)
        return y