import numpy as np
import torch,math
from PIL import Image
import torchvision
from easydict import EasyDict as edict

import torch.nn.functional as F
import torch.nn as nn
import random
from einops import repeat, rearrange

from source.rendering.point_sampler import perturb_points_per_ray
from source.rendering.aabb import intersect_aabb_end
from source.rendering.transform_perspective import compose_rotmat


def get_normal_coord(W, H, device='cpu'):
    '''
    Standard equirectangular panorama coordinate normalization.
    W: panorama width
    H: panorama height
    device: target device, usually `cpu` or `cuda`
    Returns:
        normalized_coords: tensor with shape (W, H, 3)
    '''
    # Create linear coordinates from 0 to W-1 and 0 to H-1.
    y = torch.linspace(0, W - 1, W, device=device)
    x = torch.linspace(0, H - 1, H, device=device)
    
    # Build the mesh grid.
    Y, X = torch.meshgrid(y, x, indexing='ij')  # Y: (W, H), X: (W, H)
    
    # Convert the grid to longitude and latitude.
    phi = -(Y / (W - 1) - 0.5) * 2 * math.pi + (math.pi / 2)         # longitude in [-pi, pi]
    theta = -(0.5 - X / (H - 1)) * math.pi           # latitude in [-pi/2, pi/2]
    
    # Compute normalized 3D coordinates.
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    
    normalized_coords = torch.stack([
        cos_theta * cos_phi,   # x axis
        sin_theta,             # y axis
        cos_theta * sin_phi    # z axis
    ], dim=2)  # Shape: (W, H, 3)
    # reshape to (H, W, 3)
    normalized_coords = normalized_coords.permute(1, 0, 2)
    
    return normalized_coords



def get_original_coord(W,H,full=True,c2w=None):
    '''
    W: width of pano
    H: height of pano
    if dataset is CVACT, ful=True, return the original coordinate of CVACT
    if dataset is CVUSA, ful=False,
    fill = False only used for CVUSA dataset
    '''
    normalized_coords = get_normal_coord(W,H)


    if c2w is None:
        RollPitchYaw = [0,0,0]
        R_c2w = compose_rotmat(RollPitchYaw[0], RollPitchYaw[1], RollPitchYaw[2])
        # to torch and then to the devidece of normalized_coords
        # if numpy, to torch
        if isinstance(R_c2w, np.ndarray):
            R_c2w = torch.from_numpy(R_c2w).to(normalized_coords.device).float()
    ray_directions = torch.einsum('ij,hwj->hwi', R_c2w, normalized_coords)
    # ray_directions = np.einsum('ij,hwj->hwi', R_c2w, normalized_coords)  # [H, W, 3]

    # Normalize ray directions by torch ops
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    # ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
    return ray_directions  

class Point_sampler_pano(torch.nn.Module):
    # designed for street view panorama image
    def __init__(self,
                pano_direction,
                sample_total_length=None,
                num_points=300,
                perturbation_strategy = 'uniform',
                aabb_strict =False,
                data_type = None,
                ):
        super().__init__()
        self.sample_total_length = np.sqrt(1.5**2+1.5**2+1.9**2)
        

        self.pano_direction = pano_direction
        self.num_points = num_points
        if not aabb_strict:
            self.sample_len = ((torch.arange(self.num_points)+1)*(self.sample_total_length/self.num_points)).cuda().float()

        self.voxel_low = -1
        self.voxel_max = 1

        self.perturbation_strategy = perturbation_strategy
        self.aabb_strict = aabb_strict

    @torch.no_grad()
    def forward(self,
                batch_size,
                position=None,
                ):
        origin_opensfm = position[:,None,None,:].cuda() # w -h z
        pano_direction = self.pano_direction[...,None].cuda() # b h w c # in opensfm coordinate
        output = edict()

        H,W = pano_direction.shape[1],pano_direction.shape[2]


        rays_world = repeat(pano_direction, '1 h w c 1 -> b h w c', b=batch_size )
        ray_origins = repeat(origin_opensfm, 'b 1 1 c -> b h w c', h=H, w=W )

        if self.aabb_strict:
            sample_total_length = intersect_aabb_end(ray_origins,rays_world,min=0,max=self.sample_total_length)
            sample_total_length = rearrange(sample_total_length, '(b h w) -> b h w 1', b=batch_size, h = H, w = W )
            output.radii_raw = (torch.arange(self.num_points)+1)[None,None,None,:].to(sample_total_length.device) * (sample_total_length/self.num_points)
        else:
            depth = self.sample_len[None,None,None,None,:]
            output.radii_raw  = repeat(depth, '1 1 1 1 k -> b h w k', b=batch_size, h = H, w = W )  # (1,h,w,1,k) -> (batch_size, h, w, k)
        output.radii =  perturb_points_per_ray(output.radii_raw,strategy=self.perturbation_strategy)
        sample_point = ray_origins.unsqueeze(-1) + rays_world.unsqueeze(-1) * output.radii.unsqueeze(-2)

        

        output.points_world =  rearrange(sample_point, 'b h w c k -> b h w k c').clone()
        output.ray_origins = ray_origins.clone()
        output.ray_origins[...,1] = -output.ray_origins[...,1]
        output.rays_world = rays_world.clone()
        output.rays_world[...,1] = -output.rays_world[...,1]
        output.points_world[...,1] = -output.points_world[...,1]
        return  output

                
        


def get_sat_ori(resolution,position_scale_factor=1):
    y_range =  (torch.arange(resolution,dtype=torch.float32,)+0.5)/(0.5*resolution)-1
    x_range = (torch.arange(resolution,dtype=torch.float32,)+0.5)/(0.5*resolution)-1
    Y,X = torch.meshgrid(y_range,x_range)
    Y = Y*position_scale_factor
    X = X*position_scale_factor
    Z = torch.ones_like(Y) # z=1 means the highest position in coordinate, in dimension 1
    xy_grid = torch.stack([X,Z,Y],dim=-1)[None,:,:]
    return xy_grid


class Point_sampler_ortho(torch.nn.Module):
    '''
    point sampler designed for ortho image,


    '''
    def __init__(self,
                 num_points,
                 resolution=256,
                 perturbation_strategy = 'uniform',
                 position_scale_factor = 1,
                 render_size = 128,
                 ):
        super().__init__()
        self.perturbation_strategy = perturbation_strategy
        # not used any more
        self.resolution = resolution
        self.sat_ori = get_sat_ori(self.resolution,position_scale_factor)[...,None]
        self.sat_dir  = torch.tensor([0,-1,0])[None,None,None,:,None]
        self.sample_total_length = 2
        self.num_points = num_points
        self.sample_len = ((torch.arange(self.num_points)+1)*(self.sample_total_length/self.num_points))
        self.render_size = render_size

    @torch.no_grad()
    def forward(self,
                batch_size,
                random_crop=True,
                crop_type=None,
                ):
        depth = self.sample_len[None,None,None,None,:].cuda().float()
        sat_dir = self.sat_dir.cuda()
        # sample_point = self.sat_ori + self.sat_dir * depth
        output = edict()
        if random_crop:
            if crop_type == 'crop':
                assert self.render_size < self.resolution
                start_h = random.randint(0,self.resolution-self.render_size-1)
                start_w = random.randint(0,self.resolution-self.render_size-1)
                output.idx = [start_h,start_w]
                sat_ori = self.sat_ori[:,start_h:start_h+self.render_size,start_w:start_w+self.render_size,:]
            elif crop_type == 'resize':
                sat_ori = rearrange(self.sat_ori,'b h w c 1 -> b c h w')
                sat_ori = F.interpolate(sat_ori,scale_factor=0.5,mode='bilinear')
                sat_ori = rearrange(sat_ori,'b c h w -> b h w c 1')
            else:
                raise NotImplementedError
        else:
            sat_ori = self.sat_ori
            self.render_size = self.resolution
            assert self.render_size == self.resolution
        sat_ori = sat_ori.cuda()
        
        # sat_ori = self.position_scale_factor * sat_ori
        # output.points_world =  repeat(grid, '1 k h w c -> b h w k c', b=batch_size)
        output.rays_world = repeat(sat_dir, '1 1 1 c 1 -> b h w c', b=batch_size,  h = self.render_size, w = self.render_size )[...,[0,2,1]]  # (1,h,w,3,1) -> (batch_size, h, w, 3)
        output.radii_raw  = repeat(depth, '1 1 1 1 k -> b h w k', b=batch_size, h = self.render_size, w = self.render_size )  # (1,h,w,1,k) -> (batch_size, h, w, k, 1)
        output.ray_origins = repeat(sat_ori, '1 h w c 1 -> b h w c',b=batch_size)[...,[0,2,1]]
        output.radii =  perturb_points_per_ray(output.radii_raw,strategy=self.perturbation_strategy)

        sample_point = sat_ori + sat_dir * output.radii.unsqueeze(-2)
        
        grid = sample_point.permute(0,4,1,2,3)[...,[0,2,1]] # has a change back, from height in the second dimension to height in the last dimension
        # grid[...,2]   = ((grid[...,2]-self.voxel_low)/(self.voxel_max-self.voxel_low))*2-1
        # grid = grid.float()
        output.points_world =  rearrange(grid, 'b k h w c -> b h w k c')
        return  output





# class RGB_Reprerenter(torch.nn.Module):
#     def __init__(self,
#                 ):
#         super().__init__()

#     def forward(self,
#                 points,
#                 image,
#                 ):
#         point_h_w = points[...,0:2].unsqueeze(2) # b, N, 1, 2
#         rgb_feature = F.grid_sample(image,point_h_w).squeeze(-1).permute(0,2,1) # b, C, N, 1 -> b, N, C
#         return rgb_feature


