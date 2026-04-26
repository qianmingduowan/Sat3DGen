import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import trimesh
from PIL import Image
from skimage import measure

from source.generator import Sat3DGen


def parse_args():
    parser = argparse.ArgumentParser(description="Run slicing-based inference on a large satellite image.")
    parser.add_argument("--satellite_img_path", type=str, required=True)
    parser.add_argument("--step_size", type=int, default=64)
    parser.add_argument("--model_path", type=str, default="qian43/Sat3DGen", help="Model path: HuggingFace repo id or local checkpoint directory.")
    parser.add_argument("--pixel_resolution", type=float, default=0.28)
    parser.add_argument("--grid_size", type=int, default=320)
    parser.add_argument("--mesh_level", type=float, default=4.5)
    parser.add_argument("--output_dir", type=str, default="./results/big_image_slice")
    return parser.parse_args()


HUGGINGFACE_REPO = "qian43/Sat3DGen"

def resolve_checkpoint_path(model_root):
    """Locate the model weights directory or fall back to HuggingFace.

    Accepts three layouts:
    1. ``model_root`` itself contains ``config.json`` (released weights).
    2. ``model_root/vqmodel_ema`` exists (training checkpoint with EMA).
    3. ``model_root/vqmodel`` exists (training checkpoint without EMA).
    4. None of the above → return HuggingFace repo id for auto-download.
    """
    model_root = Path(model_root)
    if model_root.name in {"vqmodel", "vqmodel_ema"}:
        raise ValueError("Please pass the checkpoint directory, not `vqmodel` or `vqmodel_ema`.")

    if (model_root / "config.json").exists():
        return str(model_root)

    ema_path = model_root / "vqmodel_ema"
    if ema_path.exists():
        return str(ema_path)

    model_path = model_root / "vqmodel"
    if model_path.exists():
        return str(model_path)

    print(f"[model] Local checkpoint not found at '{model_root}', will load from HuggingFace: {HUGGINGFACE_REPO}")
    return HUGGINGFACE_REPO


def prepare_image(image, pixel_resolution):
    if pixel_resolution != 0.28:
        factor = pixel_resolution / 0.28
        new_size = (int(image.size[0] * factor), int(image.size[1] * factor))
        print(f"Resize input image from {image.size} to {new_size}")
        image = image.resize(new_size, Image.BICUBIC)

    if min(image.size) > 1000:
        left = int(image.size[0] / 2 - 500)
        top = int(image.size[1] / 2 - 500)
        image = image.crop((left, top, left + 1000, top + 1000))

    process_size = min(image.size)
    left = (image.size[0] - process_size) // 2
    top = (image.size[1] - process_size) // 2
    return image.crop((left, top, left + process_size, top + process_size))


def build_transform(model):
    patch_size = model.unet_model.patch_size
    return T.Compose(
        [
            T.Resize((patch_size * 16, patch_size * 16), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def iter_tiles(process_size, output_size, step_size):
    all_step = ((process_size - output_size) // step_size) + 1
    if (process_size - output_size) % step_size != 0:
        all_step += 1

    w_index = 0
    for i in range(all_step):
        h_index = 0
        for j in range(all_step):
            yield w_index, h_index
            if j == all_step - 2:
                h_index = process_size - output_size
            else:
                h_index += step_size
        if i == all_step - 2:
            w_index = process_size - output_size
        else:
            w_index += step_size


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(args.model_path)

    Sat3DGen._skip_backbone_weights = True
    model = Sat3DGen.from_pretrained(checkpoint_path).to(device)
    Sat3DGen._skip_backbone_weights = False
    model.eval()

    output_root = Path(args.output_dir) / Path(args.satellite_img_path).stem / Path(args.model_path.rstrip("/")).name
    output_root.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.satellite_img_path).convert("RGB")
    image = prepare_image(image, args.pixel_resolution)
    image.save(output_root / "crop.png")

    transform = build_transform(model)
    process_size = image.size[0]
    output_size = 256
    crop_edge = 32
    height_shape = 288

    output_volume = np.zeros((process_size, process_size, height_shape), dtype=np.float32)
    output_count = np.zeros((process_size, process_size), dtype=np.float32)

    for w_index, h_index in iter_tiles(process_size, output_size, args.step_size):
        tile = image.crop((w_index, h_index, w_index + output_size, h_index + output_size))
        sat_input = transform(tile).unsqueeze(0).to(device)
        with torch.no_grad():
            density = model.save_shape_from_sat(sat_input, position_scale_factor=1, crop=True, grid_size=args.grid_size)

        density = np.moveaxis(density, 0, 1)
        crop_edge_h_st = 0 if h_index == 0 else crop_edge
        crop_edge_h_en = 0 if h_index + output_size == process_size else crop_edge
        crop_edge_w_st = 0 if w_index == 0 else crop_edge
        crop_edge_w_en = 0 if w_index + output_size == process_size else crop_edge

        output_volume[
            h_index + crop_edge_h_st : h_index + output_size - crop_edge_h_en,
            w_index + crop_edge_w_st : w_index + output_size - crop_edge_w_en,
        ] += density[
            crop_edge_h_st : density.shape[0] - crop_edge_h_en,
            crop_edge_w_st : density.shape[1] - crop_edge_w_en,
        ]
        output_count[
            h_index + crop_edge_h_st : h_index + output_size - crop_edge_h_en,
            w_index + crop_edge_w_st : w_index + output_size - crop_edge_w_en,
        ] += 1

    output_count[output_count == 0] = 1
    output_volume = output_volume / np.expand_dims(output_count, axis=-1)
    np.savez_compressed(output_root / "density_volume.npz", density=output_volume)

    verts, faces, _, _ = measure.marching_cubes(output_volume, level=args.mesh_level)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(output_root / "mesh.ply")

    print(f"Saved crop image, density volume, and mesh to {output_root}")
