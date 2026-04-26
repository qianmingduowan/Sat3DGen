import numpy as np
from scipy.spatial.transform import Rotation
import torch
from einops import repeat, rearrange
from easydict import EasyDict as edict
import torch.nn.functional as F

from source.rendering.aabb import intersect_aabb_end
from source.rendering.point_sampler import perturb_points_per_ray

def decompose_rotmat(R_c2w):
    R_cv2xyz = Rotation.from_euler("X", -90, degrees=True)
    rot_w2c = R_cv2xyz * Rotation.from_matrix(R_c2w).inv()
    roll, pitch, yaw = rot_w2c.as_euler("YXZ", degrees=True)
    return roll, pitch, yaw 

def normalize_angles(angles):
    """Normalize angles to be within the range [-180, 180] degrees."""
    return (np.array(angles) + 180) % 360 - 180

def compose_rotmat(roll, pitch, yaw):
    R_cv2xyz = Rotation.from_euler("X", -90, degrees=True)
    rot_c2w = Rotation.from_euler("YXZ", [roll, pitch, yaw], degrees=True)
    
    rot_w2c = R_cv2xyz.inv() * rot_c2w

    return rot_w2c.inv().as_matrix()


def fov_size2intrinsics(fov, img_size):
    """Converts field of view size to camera intrinsics."""
    if isinstance(fov, (int, float)):
        fov = [fov, fov]
    fov_x = np.deg2rad(fov[0])
    fov_y = np.deg2rad(fov[1])
    fx = (img_size[0] / 2) / np.tan(fov_x / 2)
    fy = (img_size[1] / 2) / np.tan(fov_y / 2)
    return np.array([[fx, 0, img_size[0] / 2],
                     [0, fy, img_size[1] / 2],
                     [0, 0, 1]])

def from_Euler_and_position_to_c2w(roll_pitch_yaw, position):
    roll, pitch, yaw = roll_pitch_yaw
    rot_c2w = compose_rotmat(roll, pitch, yaw)
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = rot_c2w
    if isinstance(position, torch.Tensor):
        position = position.cpu().numpy()
    T_c2w[:3, 3] = position
    return T_c2w

class PointSamplerPerspective(torch.nn.Module):
    def __init__(self, num_points,aabb_strict=True,perturbation_strategy = 'uniform',render_size=[128,128]):
        super().__init__()
        """
        render_size: [H,W]
        num_points: number of points to sample along each ray
        aabb_strict: whether to use strict AABB for sampling
        perturbation_strategy: strategy for perturbing points along the ray
        """
        self.aabb_strict = aabb_strict,
        self.sample_total_length = np.sqrt(1.5**2+1.5**2+1.9**2)
        self.num_points = num_points
        self.perturbation_strategy = perturbation_strategy
        self.render_size = render_size
        

    @torch.no_grad()
    def forward(self, intrinsics, c2w):
        # c2w:  B x 4 x 4
        # intrinsics: B x 3 x 3
        # return: 
        # output = edict()
        # output.rays_world: B x H x W x C # direction of the rays
        # output.radii_raw: B x H x W x K
        # output.radii: B x H x W x K
        # output.ray_origins: B x H x W x C # origin of the rays
        # output.points_world: B x H x W x K x C
        
        batch_size = c2w.shape[0]
        t = c2w[:, :3, 3].clone()
        output = edict()
        output.ray_origins = repeat(t, 'b c -> b h w c', h=self.render_size[0] , w=self.render_size[1]).cuda()
        output.ray_origins = output.ray_origins.clone()  # w -h z
        output.rays_world = compute_ray_directions(c2w.cuda(), intrinsics.cuda(), self.render_size[0], self.render_size[1])

        if self.aabb_strict:
            #  from b c to (b h w) c
            # origin_for_aabb = repeat(output.ray_origins, 'b c -> b h w c', h = H, w = W)
            # from  b h w c to (b h w) c
            # pano_direction_for_aabb = repeat(output.rays_world, 'b h w c -> b h w c', h = H, w = W)
            sample_total_length = intersect_aabb_end(output.ray_origins,output.rays_world,min=0,max=self.sample_total_length)
            sample_total_length = rearrange(sample_total_length, '(b h w) -> b h w 1', b=batch_size, h = self.render_size[0], w = self.render_size[1] )
            output.radii_raw = (torch.arange(self.num_points)+1)[None,None,None,:].to(sample_total_length.device) * (sample_total_length/self.num_points)
        else:
            raise NotImplementedError
        output.radii =  perturb_points_per_ray(output.radii_raw,strategy=self.perturbation_strategy)
        sample_point = output.ray_origins.unsqueeze(-1) + output.rays_world.unsqueeze(-1) * output.radii.unsqueeze(-2)
        output.points_world =  rearrange(sample_point, 'b h w c k -> b h w k c')
        # process_from w -h z to w h z 
        output.ray_origins[...,1] = -output.ray_origins[...,1]
        output.rays_world[...,1] = -output.rays_world[...,1]
        output.points_world[...,1] = -output.points_world[...,1]
        return output


        # return  output
    


def generate_pixel_coordinates(H, W):
    """
    Generate pixel coordinates grid on the image plane.

    Parameters:
    - H: Image height
    - W: Image width

    Returns:
    - pixel_coords: Pixel coordinates grid with shape [H, W, 3]
    """
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    # to current device
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().to(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    return pixel_coords

def compute_ray_directions(camera2world, intrinsics, H, W):
    """
    Compute ray directions.

    Parameters:
    - camera2world: Camera-to-world transformation matrix with shape [B, 4, 4]
    - intrinsics: Intrinsic matrix with shape [B, n, n]

    Returns:
    - ray_directions: Ray directions with shape [B, H, W, 3]
    """
    B = camera2world.shape[0]
    pixel_coords = generate_pixel_coordinates(H, W)  # [H, W, 3]
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 3]

    inv_intrinsics = torch.inverse(intrinsics)  # [B, n, n]
    normalized_coords = torch.einsum('bij,bhwj->bhwi', inv_intrinsics, pixel_coords)  # [B, H, W, 3]

    ray_directions = torch.einsum('bij,bhwj->bhwi', camera2world[:, :3, :3], normalized_coords)  # [B, H, W, 3]

    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    return ray_directions
