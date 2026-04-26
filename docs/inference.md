# Inference

## Inference vs Evaluation

`inference/` scripts are used to **generate outputs** from a trained checkpoint.

The released inference path assumes a CUDA-enabled environment.

Typical inference outputs are:

- a custom trajectory video
- a single panorama rendering
- a reference sky rendering
- satellite RGB and depth outputs
- a mesh export

If you want **dataset-level metrics** or **DSM-related quantitative evaluation**, use [evaluation.md](evaluation.md) instead.

## Model Weights

The model is hosted on **HuggingFace**: [qian43/Sat3DGen](https://huggingface.co/qian43/Sat3DGen)

All inference scripts default to loading from HuggingFace automatically (no
manual download required). The first run downloads the weights (~1.5 GB) and
caches them under `~/.cache/huggingface/hub/`. Subsequent runs load from cache.

The released checkpoint already **bundles the DINOv3 backbone weights**, so
there is no need to download DINOv3 separately for inference.

To use local checkpoints instead, pass `--model_path checkpoints` (or the
path to your local checkpoint directory).

## Quick Demo (One Command)

For an end-to-end demo on a single satellite image, use the wrapper script:

```bash
bash inference.sh <satellite_image_path> [gpu_id] [sky_reference_path]
```

Example:

```bash
bash inference.sh data/vigor/Seattle/satellite/satellite_xxx.png 0
```

The script will:

1. Generate a 3D triplane representation and export a textured mesh
   (`mesh.obj`).
2. If no trajectory file exists at
   `results/demo/<image_stem>/trajectory.csv`, the script will pause and ask
   you to draw one. Two ways to draw a trajectory are supported:
   - **With graphical display**: run
     `python inference/make_trajectory.py --input_img_path <image>`
     (X11 forwarding works for remote servers via `ssh -X`).
   - **Without graphical display (recommended in VSCode)**: open
     `inference/make_trajectory.ipynb`, run all cells, hold the left mouse
     button on the satellite image and drag to draw a path. The notebook
     reads parameters from `results/demo/.traj_config.json` (written by
     `inference.sh`) and saves to `trajectory.csv` automatically.
     First-time setup: `pip install ipympl`.
3. Render panorama and 4-direction perspective views along the trajectory.
4. Render an orbiting view of the 3D mesh.
5. Compose all panels into a single demo video.

All outputs are saved under `results/demo/<image_stem>/`:

| File | Description |
|---|---|
| `input_sat.png` | Input satellite image |
| `trajectory.csv` | Trajectory used for rendering |
| `trajectory.png` | Visualization of the drawn trajectory |
| `mesh.obj` | Extracted 3D mesh |
| `trajectory_video.mp4` | Satellite image + moving camera marker |
| `mesh_orbit_video.mp4` | Orbiting view of the 3D mesh |
| `panorama_video.mp4` | Panorama rendering along the trajectory |
| `streetview_video.mp4` | 4 perspective views along the trajectory |
| `demo_video.mp4` | Final composed video |

## Build A Trajectory (Standalone)

If you only want to draw a trajectory without running the full demo:

```bash
python inference/make_trajectory.py \
  --input_img_path path/to/satellite.png \
  --work_dir work_dirs/visualize_result
```

This command creates:

- `pixels.csv`
- `trajectory.png`

`pixels.csv` stores the columns `w,h,angle` and is used by
`single_image_inference.py` when `--save_video` is enabled.

Notes:

- This CLI tool requires a graphical display.
- On a remote server without display, use the notebook variant
  (`inference/make_trajectory.ipynb`) instead.
- The interaction logic is the same as the `Sat2Densitypp` trajectory tool.

## Single-Image Inference

### Common Usage

```bash
python inference/single_image_inference.py \
  --sat_img_path path/to/satellite.png \
  --data_root YOUR_VIGOR_ROOT \
  --position_path path/to/pixels.csv \
  --save_video \
  --save_sat
```

### Using A Reference Illumination Panorama

If you want to control the sky appearance or illumination reference explicitly, you can provide a reference panorama directly through `--sky_path`:

```bash
python inference/single_image_inference.py \
  --sat_img_path path/to/satellite.png \
  --sky_path path/to/reference_panorama.jpg \
  --save_street \
  --save_sky
```

If `--sky_path` is not provided, the script looks up the matching panorama from the VIGOR split metadata under `--data_root` or `--split_txt`.

### What Single-Image Inference Can Do

`inference/single_image_inference.py` supports several practical modes:

- export only the mesh
- save satellite RGB and satellite depth outputs
- render one reference panorama view
- render the inferred sky appearance
- render a full trajectory video
- render perspective-view video panels together with panorama outputs

### Required Argument Combinations

#### 1. Mesh Only

```bash
python inference/single_image_inference.py \
  --sat_img_path path/to/satellite.png \
  --only_save_mesh True
```

Required:

- `--sat_img_path`

#### 2. Save Satellite Outputs

```bash
python inference/single_image_inference.py \
  --sat_img_path path/to/satellite.png \
  --save_sat
```

Required:

- `--sat_img_path`

#### 3. Save A Reference Street View Or Sky

```bash
python inference/single_image_inference.py \
  --sat_img_path path/to/satellite.png \
  --sky_path path/to/reference_panorama.jpg \
  --save_street \
  --save_sky
```

Required:

- `--sat_img_path`
- one illumination source:
  `--sky_path`, or `--data_root` / `--split_txt`

#### 4. Render A Trajectory Video

```bash
python inference/single_image_inference.py \
  --sat_img_path path/to/satellite.png \
  --position_path path/to/pixels.csv \
  --sky_path path/to/reference_panorama.jpg \
  --save_video \
  --save_sat
```

Required:

- `--sat_img_path`
- `--position_path`
- one illumination source:
  `--sky_path`, or `--data_root` / `--split_txt`

### Main Arguments

- `--sat_img_path`: input satellite image.
- `--model_path`: model path. Defaults to `qian43/Sat3DGen` (HuggingFace). Can also be a local checkpoint directory.
- `--data_root`: prepared VIGOR root. Needed when the script has to resolve metadata automatically.
- `--split_txt`: optional explicit split file. Useful when you want to override the default split lookup.
- `--sky_path`: optional panorama used as the illumination reference.
- `--position_path`: trajectory CSV generated by `make_trajectory.py`. Required only when `--save_video` is enabled.
- `--save_video`: render a video along the provided trajectory.
- `--save_sat`: save the satellite RGB and depth outputs.
- `--save_street`: save one reference panorama rendering.
- `--save_sky`: save the inferred sky image.
- `--only_save_mesh`: export only the mesh and stop.
- `--visual_direction`: overlay direction markers on saved video frames.
- `--render_size`: optional render-size override.

### Main Outputs

Depending on the flags, the script may produce:

- `pred_satrgb.png`
- `pred_satdep.png`
- `sat_image.png`
- `sat_input.png`
- `pred_sky.png`
- `pred_street.png`
- `gt_sky_rgb.png`
- `sky_feature.png`
- `mesh_ori.obj`
- `vid.mp4`
- `vid.gif`
- `vid_str_per.mp4`
- `vid_str_raw.mp4`
- frame folders such as `save_vid/`, `save_sat/`, and `save_street_only_vid/`

Output behavior summary:

- `--only_save_mesh` exports `mesh_ori.obj` and stops early.
- `--save_sat` exports satellite RGB and depth results.
- `--save_street` exports one reference panorama rendering.
- `--save_sky` exports the inferred sky image and the reference sky RGB when available.
- `--save_video` exports trajectory frames and final videos. This mode requires `--position_path`.

## Large-Image Slicing

```bash
python inference/big_image_slice_inference.py \
  --satellite_img_path path/to/large_satellite_image.png
```

This script is kept for large satellite image slicing inference.

## DSM Visualization

```bash
python inference/visualize_dsm.py \
  --input_path output_remote/save_result/vigor/test/your_model/checkpoint-xxxxxx/DSM_result
```

Use `--invert` if you want the color map reversed for a specific DSM convention.
