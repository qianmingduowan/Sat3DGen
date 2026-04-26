import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision
import torchvision.transforms as T
import trimesh
from einops import rearrange
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage

import sys,os
sys.path.append(os.getcwd())
from source.generator import Sat3DGen
from source.rendering.transform_perspective import compose_rotmat


DEFAULT_SPLIT_FILES = [
    "train__corrected_all_3city_remove_building.txt",
    "test_remove_building.txt",
    "val__corrected_remove_building.txt",
    "val__corrected.txt",
    "val.txt",
]
TO_PIL = ToPILImage()


def check_watertight(mesh: trimesh.Trimesh) -> Tuple[bool, str]:
    if mesh.is_watertight:
        report = "The mesh is watertight."
        print(report)
        return True, report

    report_lines = ["The mesh is not watertight. Possible reasons:"]
    if not mesh.is_empty:
        num_holes = len(mesh.boundary.discrete)
        if num_holes > 0:
            report_lines.append(f" - Found {num_holes} open boundary loops.")

    nonmanifold_edges = mesh.nonmanifold_edges
    if len(nonmanifold_edges) > 0:
        report_lines.append(f" - Found {len(nonmanifold_edges)} non-manifold edges.")

    if len(report_lines) == 1:
        report_lines.append(" - No obvious holes or non-manifold edges were detected.")

    report = "\n".join(report_lines)
    print(report)
    return False, report


def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):
    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pointnp_px3)
    mesh.triangles = o3d.utility.Vector3iVector(facenp_fx3)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colornp_px3 / 255.0)

    o3d.io.write_triangle_mesh(fpath, mesh, write_vertex_normals=False)
    print(f"Mesh saved to {fpath}")


def export_mesh(model, mesh_path, planes, w_sky=None):
    with torch.no_grad():
        vertices, faces, vertex_colors = model.extract_mesh(planes, w_sky=w_sky)
    vertices = vertices[:, [1, 2, 0]]
    save_obj(vertices, faces, vertex_colors, mesh_path)


def make_histo(grd_img_path, device):
    grd_img = Image.open(grd_img_path).resize((512, 128))
    grd_img = torchvision.transforms.ToTensor()(grd_img).unsqueeze(0).float().to(device)

    if "streetview" in grd_img_path:
        mask_img_path = grd_img_path.replace("streetview", "pano_sky_mask")
    elif "panorama" in grd_img_path:
        mask_img_path = grd_img_path.replace("panorama", "pano_sky_mask")
        if mask_img_path.endswith(".jpg"):
            mask_img_path = mask_img_path.replace(".jpg", ".png")
    else:
        raise ValueError(f"Cannot infer sky-mask path from {grd_img_path}")

    mask_img = Image.open(mask_img_path).convert("L").resize((512, 128), Image.NEAREST)
    mask_img = T.ToTensor()(mask_img).unsqueeze(0).float().to(device)

    sky_image = (grd_img * mask_img).mul(2).sub(1)
    sky_image = sky_image.detach().cpu().numpy()
    from source.sky_histogram import compute_sky_histogram
    histo_sky = torch.from_numpy(
        compute_sky_histogram(sky_image[0], hist_range=(-1, 1))
    ).unsqueeze(0).float().to(device)
    return histo_sky


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sat3DGen single-image inference on VIGOR.")
    parser.add_argument("--sat_img_path", type=str, required=True, help="Path to the satellite image.")
    parser.add_argument("--model_path", type=str, default="qian43/Sat3DGen", help="Model path: HuggingFace repo id or local checkpoint directory.")
    parser.add_argument("--position_path", type=str, default=None, help="CSV trajectory file produced by make_trajectory.py.")
    parser.add_argument("--data_root", type=str, default=None, help="Prepared VIGOR dataset root.")
    parser.add_argument(
        "--split_txt",
        "--txt_file",
        dest="split_txt",
        type=str,
        default=None,
        help="Optional explicit split file. When omitted, known VIGOR split files are searched.",
    )
    parser.add_argument("--sky_path", type=str, default=None, help="Optional panorama path used to extract sky appearance.")
    parser.add_argument("--save_video", action="store_true", help="Render a video along the provided trajectory.")
    parser.add_argument("--save_sky", action="store_true", help="Save the inferred sky image.")
    parser.add_argument("--save_sat", action="store_true", help="Save satellite RGB and depth outputs.")
    parser.add_argument("--save_street", action="store_true", help="Save one reference street-view rendering.")
    parser.add_argument("--work_dir", type=str, default="./results", help="Output directory.")
    parser.add_argument("--visual_direction", action="store_true", help="Overlay direction hints on saved frames.")
    parser.add_argument("--render_size", type=int, default=None, help="Optional render size override.")
    parser.add_argument("--only_save_mesh", type=str2bool, default=False, help="Export only the mesh.")
    return parser.parse_args()


HUGGINGFACE_REPO = "qian43/Sat3DGen"

def resolve_checkpoint_path(model_root: str) -> str:
    """Locate the model weights directory or fall back to HuggingFace.

    Accepts three layouts:
    1. ``model_root`` itself contains ``config.json`` (released weights).
    2. ``model_root/vqmodel_ema`` exists (training checkpoint with EMA).
    3. ``model_root/vqmodel`` exists (training checkpoint without EMA).
    4. None of the above → return HuggingFace repo id for auto-download.
    """
    model_root_path = Path(model_root)
    if model_root_path.name in {"vqmodel", "vqmodel_ema"}:
        raise ValueError("Please pass the checkpoint directory, not `vqmodel` or `vqmodel_ema`.")

    # Released weights: config.json lives directly under model_root
    if (model_root_path / "config.json").exists():
        return str(model_root_path)

    ema_path = model_root_path / "vqmodel_ema"
    if ema_path.exists():
        return str(ema_path)

    model_path = model_root_path / "vqmodel"
    if model_path.exists():
        return str(model_path)

    print(f"[model] Local checkpoint not found at '{model_root}', will load from HuggingFace: {HUGGINGFACE_REPO}")
    return HUGGINGFACE_REPO


def resolve_output_dir(work_dir: str, sat_img_path: str, model_path: str) -> Path:
    model_root = Path(model_path.rstrip("/"))
    return Path(work_dir) / Path(sat_img_path).stem / model_root.parent.name / model_root.name


def iter_split_paths(data_root: Optional[str], split_txt: Optional[str]) -> Iterable[Path]:
    if split_txt is not None:
        split_path = Path(split_txt)
        if split_path.is_absolute() or split_path.exists():
            yield split_path
            return
        if data_root is None:
            raise ValueError("`--data_root` is required when `--split_txt` is relative.")
        yield Path(data_root) / split_txt
        return

    if data_root is None:
        return

    for name in DEFAULT_SPLIT_FILES:
        candidate = Path(data_root) / name
        if candidate.exists():
            yield candidate


def find_split_entry(sat_img_path: str, data_root: Optional[str], split_txt: Optional[str]) -> Optional[List[str]]:
    sat_name = Path(sat_img_path).name
    for split_path in iter_split_paths(data_root, split_txt):
        if not split_path.exists():
            continue
        with split_path.open("r") as handle:
            for line in handle:
                tokens = line.strip().split()
                if tokens and Path(tokens[0]).name == sat_name:
                    return tokens
    return None


def build_reference_position(tokens: Optional[List[str]], half_pixel_num: int) -> torch.Tensor:
    if tokens is None or len(tokens) < 5:
        position = np.array([0.0, 0.0, -0.85], dtype=np.float32)
    else:
        position = np.array([float(tokens[3]) / half_pixel_num, float(tokens[4]) / half_pixel_num, -0.85], dtype=np.float32)
    return torch.from_numpy(position)


def resolve_sky_path(args, sat_img_path: str) -> Tuple[str, torch.Tensor]:
    sat_name = Path(sat_img_path).name
    half_pixel_num = Image.open(sat_img_path).size[0] // 2
    split_tokens = find_split_entry(sat_img_path, args.data_root, args.split_txt)

    if args.sky_path is not None:
        sky_path = args.sky_path
    else:
        if split_tokens is None:
            raise FileNotFoundError(
                f"Could not find metadata for {sat_name}. Provide `--sky_path` or point `--data_root` / `--split_txt` to the split files."
            )
        if args.data_root is None:
            raise ValueError("`--data_root` is required when `--sky_path` is not provided.")
        sky_path = str(Path(args.data_root) / split_tokens[1])

    return sky_path, build_reference_position(split_tokens, half_pixel_num)


def read_trajectory(csv_path: str, half_pixel_num: int) -> List[Tuple[float, float, float]]:
    positions = []
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x = float(row["h"])
            y = float(row["w"])
            positions.append(((x - half_pixel_num) / half_pixel_num, (y - half_pixel_num) / half_pixel_num, -0.85))
    return positions


def position_to_c2w(position: Tuple[float, float, float], device: torch.device) -> torch.Tensor:
    rotation = compose_rotmat(0, 0, 0)
    position_array = np.array(position, dtype=np.float32)
    position_array[0] *= -1
    position_array = position_array[[1, 0, 2]]

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(rotation, dtype=np.float32)
    c2w[:3, 3] = position_array
    return torch.from_numpy(c2w).unsqueeze(0).to(device)


def build_intrinsics(device: torch.device) -> torch.Tensor:
    fovx = 120
    fovy = 120
    fx = 0.5 * 256 / np.tan(0.5 * fovx / 180.0 * np.pi)
    fy = 0.5 * 256 / np.tan(0.5 * fovy / 180.0 * np.pi)
    cx = (256 - 1) / 2.0
    cy = (256 - 1) / 2.0
    intrinsics_resize = np.array(
        [
            [fx / 2, 0, cx / 2],
            [0, fy / 2, cy / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(intrinsics_resize).unsqueeze(0).to(device)


def tensor_rgb_to_bgr(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0, 1)
    if image.dim() == 4:
        image = image.squeeze(0)
    image_np = image.permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


def tensor_depth_to_color(depth: torch.Tensor, alpha: Optional[torch.Tensor] = None, max_depth: Optional[float] = None, upscale: bool = False) -> np.ndarray:
    depth = depth.detach().cpu()
    depth_max = max_depth if max_depth is not None else depth.max().item()
    depth_max = max(depth_max, 1e-6)
    depth_uint8 = depth.div(depth_max).clamp(0, 1).mul(255).squeeze().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    if alpha is not None:
        opacity = alpha.squeeze().detach().cpu().numpy() < 0.5
        depth_color[opacity] = [0, 0, 0]
    if upscale:
        depth_color = cv2.resize(depth_color, (depth_color.shape[1] * 2, depth_color.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    return depth_color


def get_pano_rgb(output) -> torch.Tensor:
    if hasattr(output.str_output, "sr_image"):
        return output.str_output.sr_image
    return output.str_output.image_raw_compo


def get_per_rgb(output) -> torch.Tensor:
    if hasattr(output.per_output, "sr_image"):
        return output.per_output.sr_image
    return output.per_output.image_raw_compo


def save_satellite_outputs(output_dir: Path, sat_img: torch.Tensor, sat_positions: List[Tuple[float, float, float]], sat_output) -> None:
    torchvision.utils.save_image(sat_output.feature_raw[:, :3], output_dir / "pred_satrgb.png")

    sat_depth = sat_output.image_depth
    sat_depth = 1 - (sat_depth - sat_depth.min()) / (2 - sat_depth.min())
    sat_depth = sat_depth.clamp(0, 1).mul(255).squeeze().cpu().numpy().astype(np.uint8)
    cv2.imwrite(str(output_dir / "pred_satdep.png"), cv2.applyColorMap(sat_depth, cv2.COLORMAP_PLASMA))

    sat_image = sat_img[0].add(1).mul(0.5)
    torchvision.utils.save_image(sat_image, output_dir / "sat_image.png")

    sat_rgba = sat_image.mul(255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    sat_rgba = cv2.cvtColor(sat_rgba, cv2.COLOR_RGB2BGRA)
    cv2.imwrite(str(output_dir / "sat_image_alpha.png"), sat_rgba)

    sat_overlay = TO_PIL(sat_image)
    draw = ImageDraw.Draw(sat_overlay)
    size = sat_image.size(1)
    for position in sat_positions:
        x = position[1] * size / 2 + size / 2
        y = position[0] * size / 2 + size / 2
        radius = 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
    torchvision.utils.save_image(torchvision.transforms.ToTensor()(sat_overlay), output_dir / "sat_input.png")


def save_single_view_outputs(output_dir: Path, result, save_sky: bool, save_street: bool) -> None:
    if save_sky and hasattr(result, "sky_img"):
        sky_img = result.sky_img.clamp(-1, 1).mul(0.5).add(0.5)
        torchvision.utils.save_image(sky_img, output_dir / "pred_sky.png")

    if save_street:
        street_rgb = tensor_rgb_to_bgr(get_pano_rgb(result))
        street_depth = tensor_depth_to_color(
            result.str_output.image_depth,
            alpha=result.str_output.alpha_raw,
            upscale=True,
        )
        cv2.imwrite(str(output_dir / "pred_street.png"), np.concatenate([street_rgb, street_depth], axis=0))


def render_video_frames(output_dir: Path, sat_img_pil: Image.Image, trajectory: List[Tuple[float, float, float]], pano_frames, per_frames, sr_factor: int, visual_direction: bool) -> None:
    save_dirs = {
        "save_vid": output_dir / "save_vid",
        "save_street_vid": output_dir / "save_street_vid",
        "save_street_only_vid": output_dir / "save_street_only_vid",
        "save_sat": output_dir / "save_sat",
        "save_street_vid_per": output_dir / "save_street_vid_per",
        "save_street_raw": output_dir / "save_street_raw",
    }
    for path in save_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    max_street_depth = max(frame["depth"].max().item() for frame in pano_frames)
    sat_size = sat_img_pil.size[0]

    for index, frame in enumerate(pano_frames):
        save_name = f"{index:03d}.png"
        street_rgb = tensor_rgb_to_bgr(frame["rgb"])
        street_raw = tensor_rgb_to_bgr(frame["raw"])
        street_panel = street_rgb

        if visual_direction:
            result_street_to_draw = Image.fromarray(cv2.cvtColor(street_panel, cv2.COLOR_BGR2RGB))
            draw_panel = ImageDraw.Draw(result_street_to_draw)
            pano_w = street_panel.shape[1]
            pano_h = street_panel.shape[0]
            draw_panel.line((pano_w // 2, 0, pano_w // 2, pano_h), fill="red", width=1)
            draw_panel.line((3 * pano_w // 4, 0, 3 * pano_w // 4, pano_h), fill="blue", width=1)
            draw_panel.line((pano_w // 4, 0, pano_w // 4, pano_h), fill="green", width=1)
            draw_panel.line((0, 0, 0, pano_h), fill="yellow", width=1)
            draw_panel.line((pano_w - 1, 0, pano_w - 1, pano_h), fill="yellow", width=1)
            street_panel = cv2.cvtColor(np.array(result_street_to_draw), cv2.COLOR_RGB2BGR)

        sat_frame = sat_img_pil.copy()
        draw_sat = ImageDraw.Draw(sat_frame)
        x = trajectory[index][1] * sat_size / 2 + sat_size / 2
        y = trajectory[index][0] * sat_size / 2 + sat_size / 2
        radius = 3
        draw_sat.ellipse((x - radius, y - radius, x + radius, y + radius), fill="green")
        line_length = 15
        draw_sat.line((x, y - line_length, x, y), fill="red", width=2)
        draw_sat.line((x - line_length, y, x, y), fill="green", width=2)
        draw_sat.line((x, y, x + line_length, y), fill="blue", width=2)
        draw_sat.line((x, y, x, y + line_length), fill="yellow", width=2)

        sat_frame_np = cv2.cvtColor(np.array(sat_frame), cv2.COLOR_RGB2BGR)
        if street_panel.shape[0] != sat_frame_np.shape[0]:
            sat_frame_np = cv2.resize(sat_frame_np, (street_panel.shape[0], street_panel.shape[0]), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(str(save_dirs["save_sat"] / save_name), sat_frame_np)
        cv2.imwrite(str(save_dirs["save_street_vid"] / save_name), street_panel)
        cv2.imwrite(str(save_dirs["save_vid"] / save_name), np.concatenate([sat_frame_np, street_panel], axis=1))
        cv2.imwrite(str(save_dirs["save_street_only_vid"] / save_name), street_rgb)
        cv2.imwrite(str(save_dirs["save_street_raw"] / save_name), street_raw)

    for frame_index in range(0, len(per_frames), 4):
        pano_index = frame_index // 4
        save_name = f"{pano_index:03d}per.png"
        views = []
        for offset in range(4):
            per_rgb = tensor_rgb_to_bgr(per_frames[frame_index + offset]["rgb"])
            if offset == 3:
                per_rgb = cv2.flip(per_rgb, 1)
            views.append(per_rgb)

        # 4 perspective views in a single horizontal row: left(-90) | front(0) | right(90) | back(180)
        save_img = np.concatenate([views[1], views[0], views[2], views[3]], axis=1)
        cv2.imwrite(str(save_dirs["save_street_vid_per"] / save_name), save_img)


def export_video_artifacts(output_dir: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("`ffmpeg` was not found. PNG frames were saved, but videos were not exported.")
        return

    commands = [
        [ffmpeg_path, "-y", "-framerate", "5", "-pattern_type", "glob", "-i", "save_vid/*.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "vid.gif"],
        [ffmpeg_path, "-y", "-framerate", "5", "-pattern_type", "glob", "-i", "save_vid/*.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "vid.mp4"],
        [ffmpeg_path, "-y", "-framerate", "5", "-pattern_type", "glob", "-i", "save_street_vid_per/*.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "vid_str_per.mp4"],
        [ffmpeg_path, "-y", "-framerate", "5", "-pattern_type", "glob", "-i", "save_street_raw/*.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "vid_str_raw.mp4"],
    ]
    for command in commands:
        subprocess.run(command, cwd=output_dir, check=True)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sat_img_path = Path(args.sat_img_path)
    output_dir = resolve_output_dir(args.work_dir, args.sat_img_path, args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(args.model_path)
    print(f"Loading model from {checkpoint_path}")

    sat_img_pil = Image.open(sat_img_path).convert("RGB")
    half_pixel_num = sat_img_pil.size[0] // 2

    # Skip redundant backbone weight download – from_pretrained will
    # overwrite all parameters from the safetensors file anyway.
    Sat3DGen._skip_backbone_weights = True
    model = Sat3DGen.from_pretrained(checkpoint_path).to(device)
    Sat3DGen._skip_backbone_weights = False
    model.eval()
    if args.render_size is not None:
        model.point_sampler_definition(args.render_size)

    patch_size = model.unet_model.patch_size if hasattr(model.unet_model, "patch_size") else 16
    sat_input_transform = T.Compose(
        [
            T.Resize((patch_size * 16, patch_size * 16), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    sat_img_display = T.Compose([T.Resize((256, 256), interpolation=Image.BICUBIC), T.ToTensor()])(sat_img_pil)
    sat_img_raw = sat_img_display.unsqueeze(0).to(device).mul(2).sub(1)
    sat_img_input = sat_input_transform(sat_img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        triplane = model.from_sat_to_triplane(sat_img_input)

    if args.only_save_mesh:
        mesh_path = output_dir / "mesh_ori.obj"
        export_mesh(model, str(mesh_path), triplane)
        mesh_file = trimesh.load(mesh_path, process=False)
        check_watertight(mesh_file)
        return

    needs_pano = args.save_video or args.save_sky or args.save_street
    sky_path = None
    reference_position = torch.tensor([0.0, 0.0, -0.85], device=device)
    w_sky = None
    sky_feature_2d = None

    if needs_pano:
        sky_path, reference_position = resolve_sky_path(args, args.sat_img_path)
        reference_position = reference_position.to(device)
        z_ill = make_histo(sky_path, device)
        with torch.no_grad():
            w_sky = model.w_sky_prepare(z_ill)
            sky_feature_2d = model.w_sky2sky_feature_2D(w_sky, z_ill)
        if sky_feature_2d is not None:
            sky_feature_preview = rearrange(sky_feature_2d, "b c w h -> b c h w").squeeze().cpu()[:3]
            torchvision.utils.save_image(sky_feature_preview, output_dir / "sky_feature.png")

    sat_positions_for_preview: List[Tuple[float, float, float]] = []

    if args.save_video:
        if args.position_path is None:
            raise ValueError("`--position_path` is required when `--save_video` is enabled.")
        sat_positions_for_preview = read_trajectory(args.position_path, half_pixel_num)
    elif needs_pano:
        sat_positions_for_preview = [tuple(reference_position.detach().cpu().numpy())]

    if args.save_sat:
        with torch.no_grad():
            sat_result = model.from_3D_to_results(triplane, syn_pano=False, syn_sat=True, random_sat_crop=False)
        save_satellite_outputs(output_dir, sat_img_raw.cpu(), sat_positions_for_preview, sat_result.sat_output)

    if needs_pano:
        reference_c2w = position_to_c2w(tuple(reference_position.detach().cpu().numpy()), device)
        reference_c2w[:, :3, 3] = reference_c2w[:, :3, 3] * model.position_scale_factor
        with torch.no_grad():
            reference_result = model.from_3D_to_results(
                triplane,
                c2w=reference_c2w,
                w_sky=w_sky,
                sky_feature_2D=sky_feature_2d,
                syn_pano=True,
            )
        save_single_view_outputs(output_dir, reference_result, args.save_sky, args.save_street)

        if sky_path is not None:
            gt_sky_rgb = Image.open(sky_path).convert("RGB")
            if model.sr_factor == 2:
                gt_sky_rgb = gt_sky_rgb.resize((512, 128))
            gt_sky_rgb.save(output_dir / "gt_sky_rgb.png")

    if not args.save_video:
        return

    intrinsics = build_intrinsics(device)
    pano_frames = []
    per_frames = []
    yaw_values = [0, -90, 90, 180]

    with torch.no_grad():
        for position in sat_positions_for_preview:
            c2w = position_to_c2w(position, device)
            c2w[:, :3, 3] = c2w[:, :3, 3] * model.position_scale_factor
            pano_result = model.from_3D_to_results(
                triplane,
                c2w=c2w,
                w_sky=w_sky,
                sky_feature_2D=sky_feature_2d,
                syn_pano=True,
            )
            pano_frames.append(
                {
                    "rgb": get_pano_rgb(pano_result),
                    "depth": pano_result.str_output.image_depth,
                    "alpha": pano_result.str_output.alpha_raw,
                    "raw": pano_result.str_output.feature_raw[:, :3],
                }
            )

            for yaw in yaw_values:
                c2w_per = c2w.clone()
                c2w_per[:, :3, :3] = torch.from_numpy(compose_rotmat(0, 0, yaw)).unsqueeze(0).to(device)
                per_result = model.from_3D_to_results(
                    triplane,
                    c2w=c2w_per,
                    w_sky=w_sky,
                    intrinsics=intrinsics,
                    sky_feature_2D=sky_feature_2d,
                    syn_pano=False,
                    syn_per=True,
                )
                per_frames.append({"rgb": get_per_rgb(per_result)})

    render_video_frames(output_dir, sat_img_pil, sat_positions_for_preview, pano_frames, per_frames, model.sr_factor, args.visual_direction)
    export_video_artifacts(output_dir)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
