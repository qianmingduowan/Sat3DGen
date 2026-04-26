import torch
import torch.nn as nn
import torch.nn.functional as F

from source.vae_hacked  import ResnetBlock

class SuperresolutionHybrid2X(nn.Module):
    def __init__(self, in_channels, out_channels,padding_mode='zeros',v2=False):
        super().__init__()
        self.out_channels = out_channels
        self.block0 = SynthesisBlockNoUp(in_channels, 128,padding_mode=padding_mode)
        self.block1 = SynthesisBlock(128, out_channels, is_last=True,padding_mode=padding_mode,v2=v2)

    def forward(self, x):
        img = x[:, :3, :, :].clone()
        if self.out_channels == 4:
            opa = x[:, -1:, :, :].unsqueeze(1)
        else:
            opa = None
        x = self.block0(x)
        x = self.block1(x,img,opa)
        return x
    
class SynthesisBlockNoUp(torch.nn.Module):
    # conv group
    def __init__(self, in_channels, out_channels,padding_mode):
        super().__init__()
        self.conv0 = ConvBlock(in_channels=in_channels, out_channels=128,padding_mode= padding_mode)
        self.conv1 = ConvBlock(in_channels=128, out_channels=out_channels,padding_mode= padding_mode)
        self.skip_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,padding_mode= padding_mode)
    
    def forward(self, x):
        input = self.skip_conv(x)
        x = self.conv0(x)
        x = self.conv1(x)
        return x+input

class SynthesisBlock(torch.nn.Module):
    # need 2x upsample
    # need toRGB layer (1x1 conv) to get 3 channels
    def __init__(self, in_channels, out_channels,padding_mode, is_last=True, v2=False):
        super().__init__()
        self.conv0 = ConvBlock(in_channels=in_channels, out_channels = 128,padding_mode= padding_mode)
        self.conv1 = ConvBlock(in_channels=128, out_channels = 128,padding_mode= padding_mode)
        self.is_last = is_last
        self.v2 = v2
        self.toRGB = nn.Conv2d(in_channels=128, out_channels= out_channels, kernel_size=1, stride=1, padding=0,padding_mode= padding_mode, bias= not self.v2)
        self.out_channels = out_channels
    
    def forward(self, x,img,opa = None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False,antialias=True)
        img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False,antialias=True)
        if opa is not None:
            opa = F.interpolate(opa, scale_factor=2, mode='nearest', align_corners=False,antialias=True)
            x = torch.cat([x,opa],dim=1)
        x = self.conv0(x)
        x = self.conv1(x)
        if self.is_last:
            x = self.toRGB(x)
            if not self.v2:
                x = torch.tanh(x)
            img = img + x[:,:3]
            # clamp
            if opa is not None:
                opa = opa + x[:,3:]
                opa = torch.clamp(opa, 0, 1)
            img = torch.clamp(img, 0, 1)
            # x = torch.sigmoid(x)
        if opa is not None:
            return img,opa
        return img
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,padding_mode, kernel_size=3, stride=1, padding=1, if_act = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2) if if_act else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
