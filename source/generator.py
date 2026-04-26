import torch
import torch.nn as nn
import torch.nn.functional as F
# from diffusers import DDIMScheduler
import numpy as np
import math

# from diffusers import StableDiffusionPipeline, DDIMScheduler
# from pytorch_lightning import LightningModule, Trainer


from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import no_init_weights

import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path
from einops import rearrange, repeat

from easydict import EasyDict as edict

from source.vae_hacked import Decoder
from source.rendering.utils import sample_importance,unify_attributes, create_voxel
from source.rendering.point_representer import PointRepresenter
from source.rendering.point_integrator import PointIntegrator
from source.rendering.sat2density_transform_eg3d import get_original_coord,Point_sampler_pano,Point_sampler_ortho
from source.rendering.transform_perspective import PointSamplerPerspective
from source.rendering.mlp_model import MLPNetwork2
from source.sr_module import SuperresolutionHybrid2X
from source.xyz2thetaphi import xyz2thetaphi

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import tqdm

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def resolve_backbone_candidates(backbone):
    env_override_map = {
        "dinov2-base": "SAT3DGEN_DINOV2_BASE_PATH",
        "dinov2-large": "SAT3DGEN_DINOV2_LARGE_PATH",
        "dinov3-large-sat": "SAT3DGEN_DINOV3_SAT_PATH",
        "dinov3-large-lvd": "SAT3DGEN_DINOV3_LVD_PATH",
    }
    default_candidate_map = {
        "dinov2-base": [
            "facebook/dinov2-base",
        ],
        "dinov2-large": [
            "facebook/dinov2-large",
        ],
        "dinov3-large-sat": [
            "facebook/dinov3-vitl16-pretrain-sat493m",
        ],
        "dinov3-large-lvd": [
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
        ],
    }
    if backbone not in default_candidate_map:
        raise NotImplementedError(f"Unsupported backbone: {backbone}")

    candidates = []
    env_override = os.environ.get(env_override_map[backbone])
    if env_override:
        candidates.append(env_override)
    candidates.extend(default_candidate_map[backbone])
    return candidates


# Built-in backbone configs so we can create the model structure without any
# network access (the gated DINOv3 repos require authentication even to fetch
# the config file).  These are architecture-only settings and never change.
_BACKBONE_CONFIGS = {
    "dinov2-base": {
        "model_type": "dinov2",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "patch_size": 14,
        "image_size": 518,
        "num_channels": 3,
        "num_register_tokens": 0,
    },
    "dinov2-large": {
        "model_type": "dinov2",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 14,
        "image_size": 518,
        "num_channels": 3,
        "num_register_tokens": 0,
    },
    "dinov3-large-sat": {
        "model_type": "dinov3_vit",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 16,
        "image_size": 224,
        "num_channels": 3,
        "num_register_tokens": 4,
        "hidden_act": "gelu",
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-05,
        "layerscale_value": 1.0,
        "key_bias": False,
        "mlp_bias": True,
        "proj_bias": True,
        "query_bias": True,
        "value_bias": True,
        "use_gated_mlp": False,
        "rope_theta": 100.0,
        "pos_embed_rescale": 2.0,
    },
    "dinov3-large-lvd": {
        "model_type": "dinov3_vit",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 16,
        "image_size": 224,
        "num_channels": 3,
        "num_register_tokens": 4,
        "hidden_act": "gelu",
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-05,
        "layerscale_value": 1.0,
        "key_bias": False,
        "mlp_bias": True,
        "proj_bias": True,
        "query_bias": True,
        "value_bias": True,
        "use_gated_mlp": False,
        "rope_theta": 100.0,
        "pos_embed_rescale": 2.0,
    },
}

def load_backbone_model(backbone, skip_weights=False):
    """Load (or create) the backbone vision model.

    When *skip_weights* is ``True`` the model structure is instantiated
    from a built-in config dict **without** any network access.  This is
    useful when the caller will overwrite all parameters later (e.g. via
    ``Sat3DGen.from_pretrained``), avoiding a redundant multi-GB
    download of the backbone checkpoint.
    """
    if skip_weights:
        if backbone not in _BACKBONE_CONFIGS:
            raise NotImplementedError(f"No built-in config for backbone: {backbone}")
        print(f"Creating backbone structure from built-in config (skip weights): {backbone}")
        config = AutoConfig.for_model(**_BACKBONE_CONFIGS[backbone])
        with no_init_weights():
            model = AutoModel.from_config(config)
        return model.eval().requires_grad_(False)

    load_errors = []
    for candidate in resolve_backbone_candidates(backbone):
        expanded_candidate = os.path.expanduser(candidate)
        resolved_candidate = expanded_candidate if Path(expanded_candidate).exists() else candidate
        try:
            print("Trying pretrained_model_name_or_path:", resolved_candidate)
            return AutoModel.from_pretrained(resolved_candidate).eval().requires_grad_(False)
        except Exception as exc:
            load_errors.append(f"{resolved_candidate}: {exc}")

    formatted_errors = "\n".join(load_errors)
    raise RuntimeError(
        f"Failed to load the backbone `{backbone}`.\n"
        f"Tried the following candidates:\n{formatted_errors}\n"
        "You can override the lookup with the corresponding SAT3DGEN_*_PATH environment variable."
    )

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_layers      = 8,        # Number of mapping layers.
        norm = True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.norm = norm

        features_list =  [z_dim] * (num_layers) + [w_dim]
        layers = []
        for idx in range(num_layers):
            layers.append(nn.Linear(features_list[idx], features_list[idx + 1]))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        # Embed, normalize, and concat inputs.
        if self.norm:
            z = normalize_2nd_moment(z.to(torch.float32))   # normalize z to sphere
        
        x = self.mapping(z)
        return x

class dino_3d_model(nn.Module):
    def __init__(self,output_ch=192,ch_mult=[1,2,4,4,4],pad = False, with_attn=True,backbone='dinov2-base',no_hidden_states=False, no_cls_token=False, skip_backbone_weights=False):
        super().__init__()
        self.dino_model = load_backbone_model(backbone, skip_weights=skip_backbone_weights)
        if backbone == 'dinov2-base':
            z_channels = 6144 if not no_cls_token else 6144//2

            self.feature_list = [3,6,9,12] if not no_hidden_states else []
            if self.feature_list == []:
                z_channels = z_channels//4
        elif backbone in ['dinov2-large',"dinov3-large-sat","dinov3-large-lvd"]:
            z_channels = 8192 if not no_cls_token else 8192//2
            self.feature_list = [6,12,18,24] if not no_hidden_states else []
            if self.feature_list == []:
                z_channels = z_channels//4
        self.backbone = backbone
        self.no_cls_token = no_cls_token
        self.decoder = Decoder(ch=128,out_ch=output_ch,ch_mult=ch_mult,num_res_blocks=2,attn_resolutions=[],z_channels=z_channels,resolution=256,in_channels=None,with_attn=with_attn)
        self.pad = pad
        self.patch_size = self.dino_model.config.to_dict()['patch_size']
        self.num_register_tokens = self.dino_model.config.to_dict()['num_register_tokens'] if 'num_register_tokens' in self.dino_model.config.to_dict().keys() else 0

    def forward(self, inputs):
        _h,_w = inputs.shape[-2:]
        assert _h == 16 * self.patch_size
        output = self.dino_model(inputs,output_hidden_states=True)
        out_put_list = []
        if self.feature_list == []:
            out_put_list.append(output.last_hidden_state)
        else:
            for i in self.feature_list:
                out_put_list.append(output.hidden_states[i])
        # a mistake, because len(output.hidden_states) is 13
        # our last feature list is 12, 
        # and last_hidden_state is layer normed output.hidden_states[-1],
        # so we should not append it to out_put_list
        x = torch.cat(out_put_list,dim=2)
        dino_feature = rearrange(x[:,1+self.num_register_tokens:], 'b (h w) c -> b c h w', h=_h//self.patch_size , w=_w//self.patch_size)
        if not self.no_cls_token:
            cls_token = x[:,0]
            dino_feature = torch.cat([dino_feature, cls_token.unsqueeze(-1).unsqueeze(-1).repeat(1,1,_h//self.patch_size,_w//self.patch_size)],dim=1) # [2, 7680, 16, 16]
        # noise = torch.randn_like(dino_feature)
        if self.pad:
            ori_size = dino_feature.size(-1)
            pad_size = ori_size*self.pad
            # make sure pad size is int
            assert pad_size == int(pad_size), 'pad_size should be int'
            pad_size = int(pad_size)
            dino_feature = F.pad(dino_feature,(pad_size,pad_size,pad_size,pad_size),'constant', 0)
        output = self.decoder(dino_feature) # 320*320 # 4 time 2x upsampling
        return output




def convert_to_easydict(d):
    if isinstance(d, dict):
        return edict({k: convert_to_easydict(v) for k, v in d.items()})
    return d

class Sat3DGen(ModelMixin, ConfigMixin):
    # When True, skip downloading pretrained backbone weights during __init__.
    # The weights will be loaded later by from_pretrained() from the full
    # checkpoint (which already contains the backbone parameters), avoiding
    # a redundant multi-GB download of the standalone backbone model.
    _skip_backbone_weights: bool = False

    @register_to_config
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # if opt is not a edict object, convert it to edict object .
        self.opt = convert_to_easydict(opt)
        if 'sr_padding_mode' not in self.opt.keys():
            self.opt.sr_padding_mode = 'zeros'
        if 'representation_type' not in self.opt.keys():
            self.opt.representation_type = 'triplane'
        self.sat_mapping_mode = 'v2' if not hasattr(self.opt.network, 'sat_mapping_mode') else self.opt.network.sat_mapping_mode
        assert self.sat_mapping_mode in ['v2'], 'sat_mapping_mode should be v1 or v2'
        self.sr_factor = 1 if not hasattr(self.opt.network, 'sr_factor') else self.opt.network.sr_factor
        self.if_w_sky_mapping = True 
        self.backbone = 'dinov2-base' if not hasattr(self.opt, 'backbone') else self.opt.backbone
        if self.if_w_sky_mapping:
            self.z_dim = 270
            self.w_dim = 512
            self.sky_mapping = MappingNetwork(self.z_dim,self.w_dim,norm=False)
        else:
            self.z_dim = 270
            self.w_dim = 270
        assert self.sr_factor  in [1,2] , 'sr_factor should be 1 or 2'
        self.image_size = self.opt.network.image_size # not used
        self.latent_size = self.opt.network.latent_size
        self.latent_channel = self.opt.network.latent_channel
        if 'pad' in self.opt.keys():
            self.pad = self.opt.pad
            self.position_scale_factor = 1 / (self.pad*2+1)
            assert self.opt.network.position_scale_factor ==1, 'position_scale_factor should be 1,not used in this version.'
        else:
            self.position_scale_factor  = self.opt.network.position_scale_factor
            self.pad = False

        color_channels = 32 if not hasattr(self.opt.network, 'color_channels') else self.opt.network.color_channels
        self.sr_module = SuperresolutionHybrid2X(color_channels, 3,padding_mode=self.opt.sr_padding_mode,v2=True)
        if self.opt.representation_type == 'triplane':
            output_ch = self.opt.network.triplane.dim*3
        elif self.opt.representation_type in ['oneplane','oneplane_multi']:
            output_ch = self.opt.network.triplane.dim*2
        self.with_sky = True
        self.sky_input_dim = 2

        if self.with_sky:
            self.sky_decoder = Decoder(ch=32,out_ch=color_channels,ch_mult=[1,2,2,4,4,4,4],num_res_blocks=2,attn_resolutions=[],z_channels=self.w_dim ,resolution=256,in_channels=None,with_attn=False,pano_pad=True)
        self.unet_model = dino_3d_model(output_ch = output_ch, 
                                        ch_mult = self.opt.network.triplane.ch_mult if hasattr(self.opt.network.triplane, 'ch_mult') else [1,2,4,4,4],
                                        pad = self.pad,
                                        with_attn = self.opt.network.with_attn if hasattr(self.opt.network, 'with_attn') else True,
                                        backbone = self.backbone,
                                        no_hidden_states=self.opt.network.no_hidden_states  if hasattr(self.opt.network, 'no_hidden_states') else False,
                                        no_cls_token=self.opt.network.no_cls_token if hasattr(self.opt.network, 'no_cls_token') else False,
                                        skip_backbone_weights=self._skip_backbone_weights,
                                        )

        self.num_importance = self.opt.network.point_sampling_kwargs.num_importance
        # delete num_importance from self.opt.network.point_sampling_kwargs.
        self.opt.network.point_sampling_kwargs.pop('num_importance')
        if self.opt.representation_type == 'oneplane':
            input_dim_mlp =  self.opt.network.triplane.dim*2
        elif self.opt.representation_type in ['triplane','oneplane_multi']:
            input_dim_mlp = self.opt.network.triplane.dim
        self.mlp = MLPNetwork2(input_dim=input_dim_mlp,
                                hidden_dim=64,
                                output_dim=color_channels,
                                style_dim=self.w_dim,
                                )

        self.point_representer = PointRepresenter(
            representation_type=self.opt.representation_type,
            triplane_axes=None,
            coordinate_scale=None,
            )
        self.point_integrator = PointIntegrator(**self.opt.network.ray_marching_kwargs)
        unused_parameter = ['max_height','origin_height','realworld_scale']
        for i in unused_parameter:
            if i in self.opt.network.point_sampling_kwargs.keys():
                self.opt.network.point_sampling_kwargs.pop(i)
        if self.sr_factor ==2:
            self.opt.render_size = 256 
        self.point_sampler_definition(self.opt.render_size if hasattr(self.opt, 'render_size') else 256)

    def point_sampler_definition(self, render_size=256):
        pano_size = np.array([render_size*2,render_size//2]) / self.sr_factor
        self.pano_direction = get_original_coord(W=int(pano_size[0]),H=int(pano_size[1]),full=True).unsqueeze(0).cuda().float() # different with the original one
        # point_sampling_kwargs.pano_direction = 
        self.point_sampler = Point_sampler_pano(pano_direction=self.pano_direction,**self.opt.network.point_sampling_kwargs)
        self.point_sampler_per = PointSamplerPerspective(num_points=self.opt.network.point_sampling_kwargs.num_points,aabb_strict=True,render_size=[render_size// self.sr_factor,render_size// self.sr_factor])
        if render_size==256 and self.sr_factor == 2:
            self.point_sampler_sat = Point_sampler_ortho(num_points=self.opt.network.point_sampling_kwargs.num_points,position_scale_factor=self.position_scale_factor,render_size=render_size// self.sr_factor)
        else:
            self.point_sampler_sat = Point_sampler_ortho(num_points=self.opt.network.point_sampling_kwargs.num_points,position_scale_factor=self.position_scale_factor,resolution=int(render_size*1.5),render_size=render_size)
        print('render size:', render_size, 'sr_factor:', self.sr_factor)
        

    def from_sat_to_triplane(self,x):
        planes_feature = self.unet_model(x)
        if self.opt.representation_type == 'triplane':
            triplane_ori = rearrange(planes_feature, 'b (n c) h w -> b n c h w',n=3)
        elif self.opt.representation_type in  ['oneplane','oneplane_multi']:
            one_plane_ori = planes_feature[:,:self.opt.network.triplane.dim]
            one_plane_ori = rearrange(one_plane_ori, 'b (n c) h w -> b n c h w',n=1)
            one_line_ori = planes_feature[:,self.opt.network.triplane.dim:]
            one_line_ori = torch.mean(one_line_ori, dim=2, keepdim=False)
            triplane_ori = [one_plane_ori,one_line_ori]
        return triplane_ori
    
    def c2w_prepare(self, c2w):
        if c2w is not None:
            c2w[:,:3, 3] = c2w[:,:3, 3] * self.position_scale_factor
        return c2w

    def w_sky_prepare(self, z_ill):
        if z_ill is not None:
            if self.if_w_sky_mapping:
                w_sky = self.sky_mapping(z_ill)
            else:
                w_sky = z_ill
        else:
            w_sky = None
        return w_sky
    
    def w_sky2sky_feature_2D(self, w_sky, z_ill=None):
        sky_feature_2D = None
        if self.with_sky and z_ill is not None:
            sky_feature_2D = repeat(w_sky, 'b c -> b c h w', h=8, w=8)
            sky_feature_2D = self.sky_decoder(sky_feature_2D)
            sky_feature_2D = torch.sigmoid(sky_feature_2D)
            # pad to full panorama width
            b,c,h,w = sky_feature_2D.shape
            zero_pad_sky = torch.zeros((b,c,h,int(w*0.8)),device=sky_feature_2D.device)
            sky_feature_2D = torch.cat([sky_feature_2D,zero_pad_sky],dim=3)
        return sky_feature_2D
    
    def from_3D_to_results(self,
                           triplane_ori,
                           c2w=None,
                           w_sky=None,
                           sky_feature_2D=None,
                           syn_sat=False,
                           random_sat_crop=True,
                           syn_pano=True,
                           syn_per=False,
                           same_histo=False,
                           intrinsics=None,
                           coordinates=None):
        results = edict()
        point_sampling_result = []
        w_list = []
        syn_sign = []
        if type(triplane_ori) is list:
            N = triplane_ori[0].shape[0]
        else:
            N = triplane_ori.shape[0]
        # triplane_ori_repeat = triplane_ori.repeat(2,1,1,1,1)


        if syn_sat:
            point_sampling_result_sat = self.point_sampler_sat(batch_size=N,random_crop=random_sat_crop,crop_type='crop')
            point_sampling_result.append(point_sampling_result_sat)
            if not same_histo:
                w_sat = torch.zeros([N,self.w_dim], device=triplane_ori.device if type(triplane_ori) is not list else triplane_ori[0].device)
            else:
                w_sat = w_sky

            w_list.append(w_sat)
            syn_sign.append('sat')

        if syn_pano:
            resize_for_pano = False
            point_sampling_result_pano = self.point_sampler(batch_size=N,position=c2w[:,:3, 3])
            if self.training:
                if point_sampling_result_pano.rays_world.size(1) != point_sampling_result_pano.rays_world.size(2):
                    resize_for_pano = True
                    # rearrange from [4, 64, 256, 3] to [4, 64*2, 128/2, 3] 
                    point_sampling_result_pano.rays_world = rearrange(point_sampling_result_pano.rays_world, 'b h (w d) c -> b (h d) w c', d=2)
                    point_sampling_result_pano.ray_origins = rearrange(point_sampling_result_pano.ray_origins, 'b h (w d) c -> b (h d) w c', d=2)
                    point_sampling_result_pano.points_world = rearrange(point_sampling_result_pano.points_world, 'b h (w d) n c -> b (h d) w n c', d=2)
                    point_sampling_result_pano.radii = rearrange(point_sampling_result_pano.radii, 'b h (w d) c -> b (h d) w c', d=2)
            point_sampling_result.append(point_sampling_result_pano)
            w_list.append(w_sky)
            syn_sign.append('pano')
        if syn_per:
            point_sampling_result_per = self.point_sampler_per(intrinsics=intrinsics, c2w=c2w)
            point_sampling_result.append(point_sampling_result_per)
            w_list.append(w_sky)
            syn_sign.append('pespective')

        if self.training and len(point_sampling_result) >1:
            point_sampling_result_cat = edict()
            point_sampling_result_cat.rays_world   = torch.cat([i.rays_world for i in point_sampling_result],dim=0)
            point_sampling_result_cat.ray_origins  = torch.cat([i.ray_origins for i in point_sampling_result],dim=0)
            point_sampling_result_cat.points_world = torch.cat([i.points_world for i in point_sampling_result],dim=0)
            point_sampling_result_cat.radii        = torch.cat([i.radii for i in point_sampling_result],dim=0)

            w_input = torch.cat(w_list,dim=0)
            if self.opt.representation_type == 'triplane':
                feature_input = triplane_ori.repeat(len(point_sampling_result),1,1,1,1)

            elif self.opt.representation_type in ['oneplane','oneplane_multi']:
                feature_input = [triplane_ori[0].repeat(len(point_sampling_result),1,1,1,1),triplane_ori[1].repeat(len(point_sampling_result),1,1)]
            output = self.from_point_sampling2result(point_sampling_result_cat,
                                            feature_input,
                                            w_sky=w_input,
                                            )
        else:
            for i in range(len(point_sampling_result)):
                if syn_sign[i] == 'sat':
                    results.sat_output = self.from_point_sampling2result(point_sampling_result[i],
                                            triplane_ori,
                                            w_sky=w_list[i],
                                            )
                elif syn_sign[i] == 'pano':
                    results.str_output = self.from_point_sampling2result(point_sampling_result[i],
                                            triplane_ori,
                                            w_sky=w_list[i],
                                            )
                elif syn_sign[i] == 'pespective':
                    results.per_output = self.from_point_sampling2result(point_sampling_result[i],
                                            triplane_ori,
                                            w_sky=w_list[i],
                                            )
        if 'sat' in syn_sign:
            if self.training and len(point_sampling_result) >1:
                results.sat_output = edict()
                results.sat_output.feature_raw = output.feature_raw[:N]
                results.sat_output.alpha_raw = output.alpha_raw[:N]
                results.sat_output.image_depth = output.image_depth[:N]
                results.sat_output.image_radii = output.image_radii[:N]
            if 'idx' in point_sampling_result_sat.keys():
                results.sat_output.idx = point_sampling_result_sat['idx']

        if 'pano' in syn_sign:
            if self.training and len(point_sampling_result) >1:
                results.str_output = edict()
                results.str_output.feature_raw = output.feature_raw[N:2*N]
                results.str_output.alpha_raw = output.alpha_raw[N:2*N]
                results.str_output.image_depth = output.image_depth[N:2*N]
                results.str_output.image_radii = output.image_radii[N:2*N]

                if resize_for_pano:
                    results.str_output.feature_raw = rearrange(results.str_output.feature_raw, 'b c (h d) w -> b c h (w d)', d=2)
                    results.str_output.alpha_raw = rearrange(results.str_output.alpha_raw, 'b c (h d) w -> b c h (w d)', d=2)
                    results.str_output.image_depth = rearrange(results.str_output.image_depth, 'b c (h d) w -> b c h (w d)', d=2)
                    results.str_output.image_radii = rearrange(results.str_output.image_radii, 'b c (h d) w -> b c h (w d)', d=2) 
            if 'idx' in point_sampling_result_pano.keys():
                results.str_output.idx = point_sampling_result_pano['idx']

            results.str_output.ray_direction = point_sampling_result_pano.rays_world
            # render sky
            if self.with_sky:
                ray_direction = xyz2thetaphi(results.str_output.ray_direction)
                sky_img = F.grid_sample(sky_feature_2D, ray_direction,align_corners=True)
                sky_img = torch.clamp(sky_img, 0, 1)
                if resize_for_pano:
                    sky_img = rearrange(sky_img, 'b c (h d) w -> b c h (w d)', d=2)
                rgb_feature_compo = results.str_output.feature_raw * results.str_output.alpha_raw + sky_img * (1 - results.str_output.alpha_raw)
                results.str_output.sky_img = sky_img
                results.str_output.image_raw_compo = rgb_feature_compo
                if self.sr_factor == 2:
                    results.str_output.sr_image = self.sr_module(rgb_feature_compo)

        if 'pespective' in syn_sign:
            if self.training and len(point_sampling_result) >1:
                results.per_output = edict()
                results.per_output.feature_raw = output.feature_raw[-N:]
                results.per_output.alpha_raw = output.alpha_raw[-N:]
                results.per_output.image_depth = output.image_depth[-N:]
                results.per_output.image_radii = output.image_radii[-N:]


            if 'idx' in point_sampling_result_per.keys():
                results.per_output.idx = point_sampling_result_per['idx']

            results.per_output.ray_direction = point_sampling_result_per.rays_world

            # render sky
            if self.with_sky:
                ray_direction = xyz2thetaphi(results.per_output.ray_direction)
                sky_img = F.grid_sample(sky_feature_2D, ray_direction,align_corners=True)
                sky_img = torch.clamp(sky_img, 0, 1)
                rgb_feature_compo = results.per_output.feature_raw * results.per_output.alpha_raw + sky_img * (1 - results.per_output.alpha_raw)
                results.per_output.sky_img = sky_img
                results.per_output.image_raw_compo = rgb_feature_compo
                if self.sr_factor == 2:
                    results.per_output.sr_image = self.sr_module(rgb_feature_compo)

        if coordinates is not None:
            # for density regularization
            results.density = self.density_reg(coordinates,triplane_ori)
        return results
    
    def density_reg(self,coordinates,triplane_ori,sample_color=False,w_sky=None):
        # Only for density regularization in training process.
        assert coordinates is not None
        sample_result = self.sample_mixed(coordinates,
                                            triplane_ori,
                                            sample_color=sample_color,
                                            w_sky=w_sky,
                                            )
        sample_density = sample_result['density']
        color_result = sample_result['color'][...,:3] if sample_color==True else None
        if self.opt.network.ray_marching_kwargs.density_clamp_mode == 'mipnerf':
            sample_density = F.softplus(sample_density - 1)
        elif self.opt.network.ray_marching_kwargs.density_clamp_mode == 'relu':
            sample_density = F.relu(sample_density + 3)
        else:
            raise NotImplementedError
        if sample_color:
            return color_result
        return sample_density


    def forward(self, 
                x,
                z_ill=None,
                syn_sat=False,
                random_sat_crop=True, 
                syn_pano=True,
                syn_per=False,
                same_histo=False,
                intrinsics=None,
                c2w=None,
                coordinates=None,
                ):
        c2w = self.c2w_prepare(c2w)

        triplane_ori = self.from_sat_to_triplane(x)

        w_sky = self.w_sky_prepare(z_ill)

        sky_feature_2D = self.w_sky2sky_feature_2D(w_sky,z_ill)


        results = self.from_3D_to_results(triplane_ori,
                                          c2w,
                                          w_sky,
                                          sky_feature_2D,
                                          syn_sat=syn_sat,
                                          random_sat_crop=random_sat_crop,
                                          syn_pano=syn_pano,
                                          syn_per=syn_per,
                                          same_histo=same_histo,
                                          intrinsics=intrinsics,
                                          coordinates=coordinates)
        results.triplane = triplane_ori

        return results
         
        




    def sample_mixed(self,
                     coordinates,
                     triplanes,
                     sample_color=False,
                     w_sky=None
                     ):



        point_features = self.point_representer(
            coordinates, ref_representation=triplanes)
        color_density_result = self.mlp(point_features,only_density=not sample_color,style=w_sky) # point_features: B N C

        return color_density_result


    
    def from_point_sampling2result(self,
                                        point_sampling_result,
                                        triplanes,
                                        w_sky=None,
                                        **synthesis_kwarg
                                          ):
        points = point_sampling_result['points_world']  # [N, H, W, K, 3]
        ray_dirs = point_sampling_result['rays_world']  # [N, H, W, 3]
        radii_coarse = point_sampling_result['radii']  # [N, H, W, K]
        ray_origins = point_sampling_result['ray_origins'] # [N, 3]

        _, H, W, K, _ = points.shape
        R = H * W
        points_coarse = rearrange(points, 'n h w k c -> n (h w) k c')  # [N, R, K, 3]
        points = rearrange(points, 'n h w k c -> n (h w k) c')  # [N, R * K, 3]
        ray_dirs = rearrange(ray_dirs, 'n h w c -> n (h w) c')
        if len(ray_origins.shape) == 4:
            ray_origins = rearrange(ray_origins, 'n h w c -> n (h w) c')
        elif len(ray_origins.shape) == 2:
            ray_origins = repeat(ray_origins, 'n c -> n (h w) c', h=R, w=1)
        radii_coarse = rearrange(radii_coarse, 'n h w k -> n (h w) k 1')

        point_features = self.point_representer(
            points, ref_representation=triplanes)  # [N, R * K, C]
        color_density_result = self.mlp(point_features,w_sky) # point_features: B N C

        densities_coarse = color_density_result['density']  # [N, R * K, 1]
        colors_coarse = color_density_result['color']  # [N, R * K, C]
        densities_coarse = rearrange(densities_coarse, 'n (r k) c -> n r k c', r=R, k=K)
        colors_coarse = rearrange(colors_coarse, 'n (r k) c -> n r k c', r=R, k=K)

        if self.num_importance > 0:
            # Do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                     densities_coarse,
                                                     radii_coarse)
            weights = rendering_result['weight']

            # Importance sampling.
            radii_fine = sample_importance(radii_coarse,
                                           weights,
                                           self.num_importance,
                                           smooth_weights=True)
            points = ray_origins.unsqueeze(
                -2) + radii_fine * ray_dirs.unsqueeze(
                -2)  # [N, R, num_importance, 3]
            points_fine = points
            points = rearrange(points, 'n r k c -> n (r k) c')  # [N, R * num_importance, 3]

            point_features = self.point_representer(
                points, ref_representation=triplanes)
            color_density_result = self.mlp(point_features,w_sky)

            densities_fine = color_density_result['density']
            colors_fine = color_density_result['color']
            densities_fine = rearrange(densities_fine, 'n (r k) c -> n r k c', r=R, k=self.num_importance)
            colors_fine = rearrange(colors_fine, 'n (r k) c -> n r k c', r=R, k=self.num_importance)

            # Gather coarse and fine results together.
            (all_radiis, all_colors, all_densities,
             all_points) = unify_attributes(radii_coarse,
                                            colors_coarse,
                                            densities_coarse,
                                            radii_fine,
                                            colors_fine,
                                            densities_fine,
                                            points1=points_coarse,
                                            points2=points_fine)

            # Do the integration along the fine pass.
            rendering_result = self.point_integrator(all_colors,
                                                     all_densities,
                                                     all_radiis)

        else:
            # Only do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                     densities_coarse,
                                                     radii_coarse)
            # all_points = points_coarse  # [N, R, K, 3]

        feature_samples = rendering_result['composite_color']
        radii_samples = rendering_result['composite_radial_dist']

        feature_image = rearrange(feature_samples, 'n (h w) c -> n c h w', h=H, w=W).contiguous()  # [N, C, H, W]
        image_radii = rearrange(radii_samples, 'n (h w) c -> n c h w', h=H, w=W).contiguous()  # [N, 1, H, W]

        image_alpha = rearrange(rendering_result['opacity'], 'n (h w) c -> n c h w', h=H, w=W).contiguous()
        image_depth = rearrange(rendering_result['composite_radial_dist'], 'n (h w) c -> n c h w', h=H, w=W).contiguous()

        # rgb_image = feature_image[:, :3]
        result = edict()
        result.feature_raw = feature_image
        result.alpha_raw = image_alpha
        # result.image_raw = rgb_image
        result.image_depth = image_depth
        result.image_radii = image_radii
        result.ray_origin = ray_origins
        if 'idx' in point_sampling_result.keys():
            result.idx = point_sampling_result['idx']
        return result

    @torch.no_grad()
    def forward_grid(self, planes, grid_size=256,position_scale_factor=1,crop=False):
        max_batch = 15000000
        # size = int(grid_size/self.position_scale_factor)
        voxel_grid = create_voxel(N=grid_size,position_scale_factor=1)['voxel_grid'].cuda()
        densities = torch.zeros(
            (voxel_grid.shape[0], voxel_grid.shape[1], 1)).cuda()
        # data/CVACT/satview_correct/

        # read img to cuda, [-1,1]

        head = 0
        with tqdm.tqdm(total=voxel_grid.shape[1]) as pbar:
            with torch.no_grad():
                while head < voxel_grid.shape[1]:
                    density = self.density_reg(coordinates=voxel_grid[:, head:head + max_batch],triplane_ori=planes)
                    # density = self.forward(sat_img,
                    #                        z,
                    #                        None,
                    #                        syn_pano=False,
                    #                        coordinates=voxel_grid[:, head:head + max_batch])['density']
                    # if self.opt.network.ray_marching_kwargs.density_clamp_mode == 'mipnerf':
                    #     densities = F.softplus(densities - 1)
                    # else:
                    #     raise NotImplementedError

                    # density = G.sample(
                    #     voxel_grid[:, head:head + max_batch],
                    #     batch_codes,
                    #     sat_img,
                    #     **G_kwargs)['density']
                    densities[:, head:head + max_batch] = density
                    head = head + max_batch
                    pbar.update(max_batch)

        densities = densities.reshape(
            (grid_size, grid_size, grid_size)).cpu().numpy()
        # densities = np.flip(densities, 0)
        # densities = np.flip(densities, 0)
        # densities = np.flip(densities, 1)
        # # Trim the border of the extracted cube.
        if self.position_scale_factor < 1:
            pad = int(np.round(((1-self.position_scale_factor)*densities.shape[0]/2)))
            if not crop:
                # densities = densities[pad:-pad, pad:-pad, pad:]
                # return densities
                # else:
                pad_value = 0
                densities[:pad] = pad_value
                densities[-pad:] = pad_value
                densities[:, :pad] = pad_value
                densities[:, -pad:] = pad_value
                densities[:, :, :pad] = pad_value # z space
            else:
                densities = densities[pad:-pad, pad:-pad, pad:]
        return densities


    @torch.no_grad()
    def save_shape_from_sat(self, sat_img, position_scale_factor=1,crop=False,grid_size=320):
        planes = self.from_sat_to_triplane(sat_img)

        return self.forward_grid(planes,position_scale_factor=1,crop=crop,grid_size=grid_size)

    @torch.no_grad()
    def save_shape(self, planes,position_scale_factor=1,save_type='density',crop=False):
        densities = self.forward_grid(planes,position_scale_factor=position_scale_factor)

        if save_type == 'density':
            try:
                import mrcfile
            except ImportError:
                raise ImportError("mrcfile is required for density export. Install via: pip install mrcfile")
            with mrcfile.new_mmap(f'0000.mrc',
                                    overwrite=True,
                                    shape=densities.shape,
                                    mrc_mode=2) as mrc:
                mrc.data[:] = densities
            print('save density done')
        
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required for 3D shape export. Install via: pip install open3d")
        if save_type == 'mesh':
            from skimage import measure
            import trimesh
            # Extract a mesh with Marching Cubes.
            verts, faces, _, _ = measure.marching_cubes(densities, level=4.5)


            # Build the Trimesh object.
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            # Compute vertex normals.
            mesh.vertex_normals

            # Optional mesh visualization.
            # mesh.show()

            # Export the mesh as a PLY file.
            mesh.export('mesh.ply')




        if save_type in  ['pointcloud','voxel']:



            def efficient_filter_numpy(densities, threshold=5):
                size = densities.shape[0]
                
                # Mark voxels whose density is above the threshold.
                high_density = np.where(densities >= threshold, 1, 0)
                
                # Count high-density voxels along each local axis.
                x_sum = high_density[:-2, 1:-1, 1:-1] +high_density[1:-1, 1:-1, 1:-1] + high_density[2:, 1:-1, 1:-1]
                y_sum = high_density[1:-1, :-2, 1:-1] + high_density[1:-1, 1:-1, 1:-1] + high_density[1:-1, 2:, 1:-1]
                z_sum = high_density[1:-1, 1:-1, :-2] + high_density[1:-1, 1:-1, 1:-1] + high_density[1:-1, 1:-1, 2:]
                # Keep only voxels that satisfy all local support conditions.
                mask = (x_sum == 3) & (y_sum == 3) & (z_sum == 3)
                
                # Remove voxels that pass the mask.
                densities[1:-1, 1:-1, 1:-1][mask] = 0
                
                return densities

            # print the number of voxels >= 5
            print('the number of voxels >= 5 before filtering:', np.sum(densities >= 5))
            densities = efficient_filter_numpy(densities)
            # print the number of voxels >= 5 after filtering
            print('the number of voxels >= 5 after filtering:', np.sum(densities >= 5))
                
                            
                        

            points = np.array(np.where(densities >= 5)).T
            points = (points / size) *2 - 1

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)



            # def position_to_color(points):
            #     # Map point coordinates from [-1, 1] to [0, 1].
            #     normalized_points = (points + 1) / 3

            #     # Use x, y, z as r, g, b.
            #     colors = normalized_points

            #     # A more complex color mapping is also possible.
            #     # colors = np.column_stack([
            #     #     normalized_points[:, 0],  # r from x
            #     #     (normalized_points[:, 1] + normalized_points[:, 2]) / 2,  # g from (y+z)/2
            #     #     1 - normalized_points[:, 2]  # b from 1-z
            #     # ])

            #     return colors
            # colors = position_to_color(np.asarray(point_cloud.points))
            # point_cloud.colors = o3d.utility.Vector3dVector(colors)


            if save_type == 'pointcloud':
                # save point cloud
                o3d.io.write_point_cloud("point_cloud.ply", point_cloud)

            if save_type == 'voxel':
                voxel_size = (1 / size)* 2 
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)
                o3d.io.write_voxel_grid("voxel_grid.ply", voxel_grid)

            # elif save_type == 'mesh':
            #     mesh = voxel_grid.to_mesh()
            #     o3d.io.write_triangle_mesh("mesh.ply", mesh)

            # print(xyz.shape)

        return 0
        
    
    def extract_mesh(
        self, 
        planes: torch.Tensor, 
        mesh_resolution: int = 320, 
        mesh_threshold: int = 5.0, 
        w_sky = None,
        **kwargs,
    ):
        '''
        Extract a 3D mesh from triplane nerf. Only support batch_size 1.
        :param planes: triplane features
        :param mesh_resolution: marching cubes resolution
        :param mesh_threshold: iso-surface threshold
        '''
        print('mesh_resolution:', mesh_resolution)
        device = planes.device if type(planes) is not list else planes[0].device

        grid_out = self.forward_grid(
            planes=planes,
            grid_size=mesh_resolution,
        )
        try:
            import mcubes
        except ImportError:
            raise ImportError("PyMCubes is required for mesh extraction. Install via: pip install PyMCubes")
        vertices, faces = mcubes.marching_cubes(
            grid_out, 
            mesh_threshold,
        )
        vertices = vertices / (mesh_resolution - 1) * 2 - 1
        # query vertex colors
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
        vertices_colors = self.density_reg(vertices_tensor,planes,sample_color=True,w_sky=w_sky)
        vertices_colors = (vertices_colors * 255).squeeze(0).cpu().numpy().astype(np.uint8)
        return vertices, faces, vertices_colors
        
        

class EMANorm(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.register_buffer('magnitude_ema', torch.ones([]))
        self.beta = beta

    def forward(self, x):
        if self.training:
            magnitude_cur = x.detach().to(torch.float32).square().mean()
            self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.beta))
        input_gain = self.magnitude_ema.rsqrt()
        x = x.mul(input_gain)
        return x


# Backward-compatible alias so that existing config.json files with
# "_class_name": "VAE_finetune" (e.g. on HuggingFace) keep working.
VAE_finetune = Sat3DGen

   
