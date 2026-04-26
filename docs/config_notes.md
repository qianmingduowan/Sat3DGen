# Config Notes

The released training config is:

- `configs/dino_v3_large_sat_0906.json`

Important fields:

- `opt.backbone`: The image encoder backbone. This release uses `dinov3-large-sat`.
- `opt.representation_type`: The 3D representation type. This release uses `oneplane`.
- `opt.pad`: Add spatial padding mode before decoding.
- `opt.network.no_hidden_states`: Uses only the final DINO features.
- `opt.network.no_cls_token`: Removes the CLS token branch from the decoder input.
- `opt.network.color_channels`: Feature width used by the renderer and super-resolution head.
- `opt.network.sr_factor`: Whether the 2x super-resolution head is enabled.
- `opt.network.point_sampling_kwargs.num_points`: Number of coarse ray samples.
- `opt.network.point_sampling_kwargs.num_importance`: Number of importance samples.
- `opt.network.ray_marching_kwargs.density_clamp_mode`: Density activation used during ray marching.
- `opt.network.triplane.dim`: Plane feature dimension.
- `opt.network.triplane.ch_mult`: Decoder channel multiplier schedule.

JSON does not support inline comments, so this file is the released parameter note for the kept config.
