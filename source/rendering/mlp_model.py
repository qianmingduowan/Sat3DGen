import torch
import torch.nn as nn
import numpy as np
from einops import repeat

class SKYMLP(nn.Module):
    r"""MLP converting ray directions to sky features."""
    def __init__(self, in_channels, style_dim, L=None,out_channels_c=3,
                 hidden_channels=256,is_pos_embedding = True):
        super(SKYMLP, self).__init__()
        self.is_pos_embedding = is_pos_embedding
        self.L = L
        self.fc_z_a = nn.Linear(style_dim, hidden_channels, bias=False)
        input_channel = in_channels+ 2*self.L*in_channels if is_pos_embedding else in_channels
        self.fc1 = nn.Linear(input_channel, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels*2, hidden_channels)
        # self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        # self.fc4 = nn.Linear(hidden_channels, hidden_channels)
        # self.fc5 = nn.Linear(hidden_channels, hidden_channels)
        self.fc_out_c = nn.Linear(hidden_channels, out_channels_c)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def positional_encoding(self,input): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(self.L,dtype=torch.float32).to(input.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
    def forward(self, x, z):
        r"""Forward network
        Args:
            x (... x in_channels tensor): Ray direction embeddings.
            z (... x style_dim tensor): Style codes.
        """
        x = torch.cat([x,self.positional_encoding(x)],dim=-1) if self.is_pos_embedding else x
        z = self.fc_z_a(z)
        assert len(x.shape) == 4
        (H,W )= x.shape[1:3]
        z = repeat(z,'b c -> b h w c',h=H,w=W)
        # z = z.repeat(1,H,W,1)
        # y = self.act(self.fc1(x) + z)
        # # cat 
        y = self.act(torch.cat([self.fc1(x),z],dim=-1))
        y = self.act(self.fc2(y))
        c = self.fc_out_c(y)
        c = torch.sigmoid(c)
        return c
    
class MLPNetwork2(nn.Module):
    """Defines fully-connected layer head in EG3D."""

    def __init__(self, input_dim, hidden_dim, output_dim,style_dim=270):
        super().__init__()

        self.net0 = nn.Linear(input_dim, hidden_dim)
        self.net0_act = nn.Softplus()
        self.net1_feature    = nn.Linear(hidden_dim, hidden_dim//2)
        self.net1_density    = nn.Linear(hidden_dim, 1)
        self.style_dim = style_dim
        self.style_squ = nn.Linear(self.style_dim,hidden_dim//2)
        self.grd_color_convert = nn.Linear(hidden_dim, output_dim)

    def forward(self, point_features, style=None, only_density=False):

        N, M, C = point_features.shape
        point_features = point_features.view(N * M, C)
        y = self.net0(point_features)
        y = self.net0_act(y)
        density = self.net1_density(y).view(N, M, -1)
        result = {}
        result['density'] = density
        if only_density:
            return result

        color = self.net1_feature(y).view(N, M, -1)
        if style is None:
            style = repeat(torch.zeros([self.style_dim]), 'd -> n m d', n=N,m=M).float().to(point_features.device)
        style = self.style_squ(style)

        if len(style.shape) == 2:
            style = repeat(style, 'n d -> n m d', m=M).float().to(point_features.device)
        combine_color_style = torch.cat([color, style], dim=-1)
        color = self.grd_color_convert(combine_color_style)

        color = torch.sigmoid(color)
        result['color'] = color
        return result
