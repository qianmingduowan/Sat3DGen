import torch
import numpy as np


def xyz2thetaphi(xyz):
    """
    xyz: (N, ..., 3) tensor
    """
    # Normalize the input tensor
    xyz = xyz / torch.norm(xyz, dim=-1, keepdim=True)

    # Calculate theta and phi 
    theta = torch.acos(xyz[..., 2]) # data range [0, pi]
    phi = torch.atan2(xyz[..., 1], xyz[..., 0]) # data range [-pi, pi]

    # to [-1,1]
    theta = (theta / 3.141592653589793) * 2 - 1
    phi = phi / 3.141592653589793

    # cat
    thetaphi = torch.cat([theta.unsqueeze(-1), phi.unsqueeze(-1)], dim=-1)

    return thetaphi