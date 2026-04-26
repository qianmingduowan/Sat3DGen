"""Sat3DGen Demo Inference Script.

Given a single satellite image, this script:
  1. Loads the model and generates a 3D triplane representation + mesh.
  2. Reads a pre-generated trajectory CSV (same-name .csv alongside the image).
  3. Renders panorama and 4-direction perspective views along the trajectory.
  4. Renders a 3D mesh orbit video (top-side circular orbit).
  5. Composes a 2x2 grid video:
       Top-left:     Satellite image with real-time camera position
       Top-right:    Panorama video
       Bottom-left:  3D mesh orbit video
       Bottom-right: 4 perspective views in a row (left, front, right, back)
  6. Saves only: input satellite image, input trajectory, final composed video.

Usage:
    python inference/demo_inference.py \
        --sat_img_path demo_images/satellite/sat_demo_6.png \
        --sky_path demo_images/panorama/pano_demo_5.jpg \
        --work_dir results/demo
"""

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as T
import trimesh
from PIL import Image, ImageDraw


import sys
import os
sys.path.append(os.getcwd())
from source.generator import Sat3DGen
from source.rendering.transform_perspective import compose_rotmat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Sat3DGen demo: single satellite image → composed video.")
    parser.add_argument("--sat_img_path", type=str, required=True, help="Path to the satellite image.")
    parser.add_argument("--model_path", type=str, default="qian43/Sat3DGen", help="Model path: HuggingFace repo id or local checkpoint directory.")
    parser.add_argument("--trajectory_path", type=str, default=None,
                        help="CSV trajectory file. If omitted, looks for a same-name .csv next to the image.")
    parser.add_argument("--sky_path", type=str, default="demo_images/panorama/pano_demo_5.jpg",
                        help="Panorama image for sky appearance conditioning.")
    parser.add_argument("--work_dir", type=str, default="./results/demo", help="Output directory.")
    parser.add_argument("--fps", type=int, default=5, help="Video frame rate.")
    parser.add_argument("--mesh_resolution", type=int, default=256, help="Mesh extraction voxel resolution.")
    parser.add_argument("--orbit_frames", type=int, default=0,
                        help="Number of orbit frames for 3D mesh video. 0 = match trajectory length.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

HUGGINGFACE_REPO = "qian43/Sat3DGen"

def resolve_checkpoint_path(model_root: str) -> str:
    """Return a local checkpoint directory or fall back to the HuggingFace repo id."""
    model_root_path = Path(model_root)
    if (model_root_path / "config.json").exists():
        return str(model_root_path)
    for sub in ("vqmodel_ema", "vqmodel"):
        candidate = model_root_path / sub
        if candidate.exists():
            return str(candidate)
    print(f"[model] Local checkpoint not found at '{model_root}', will load from HuggingFace: {HUGGINGFACE_REPO}")
    return HUGGINGFACE_REPO


# ---------------------------------------------------------------------------
# Sky histogram
# ---------------------------------------------------------------------------

def make_histo(grd_img_path: str, device: torch.device) -> torch.Tensor:
    grd_img = Image.open(grd_img_path).convert("RGB").resize((512, 128))
    grd_tensor = T.ToTensor()(grd_img).unsqueeze(0).float().to(device)

    grd_path = Path(grd_img_path)
    parent_name = grd_path.parent.name
    if parent_name in ("streetview", "panorama"):
        mask_dir = grd_path.parent.parent / "pano_sky_mask"
        mask_img_path = str(mask_dir / grd_path.with_suffix(".png").name)
    else:
        raise ValueError(f"Cannot infer sky-mask path from {grd_img_path}")

    mask_img = Image.open(mask_img_path).convert("L").resize((512, 128), Image.NEAREST)
    mask_tensor = T.ToTensor()(mask_img).unsqueeze(0).float().to(device)

    sky_image = (grd_tensor * mask_tensor).mul(2).sub(1).detach().cpu().numpy()
    from source.sky_histogram import compute_sky_histogram
    histo = torch.from_numpy(
        compute_sky_histogram(sky_image[0], hist_range=(-1, 1))
    ).unsqueeze(0).float().to(device)
    return histo


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------

def read_trajectory_csv(csv_path: str, sat_size: int) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
    """Read trajectory CSV (w,h,angle) and return normalized positions + pixel coords."""
    half = sat_size / 2.0
    positions = []
    pixel_coords = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            px = float(row["w"])
            py = float(row["h"])
            pixel_coords.append((px, py))
            positions.append(((py - half) / half, (px - half) / half, -0.85))
    return positions, np.array(pixel_coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def position_to_c2w(position: Tuple[float, float, float], device: torch.device) -> torch.Tensor:
    rotation = compose_rotmat(0, 0, 0)
    pos = np.array(position, dtype=np.float32)
    pos[0] *= -1
    pos = pos[[1, 0, 2]]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(rotation, dtype=np.float32)
    c2w[:3, 3] = pos
    return torch.from_numpy(c2w).unsqueeze(0).to(device)


def build_intrinsics(device: torch.device) -> torch.Tensor:
    fovx, fovy = 120, 120
    fx = 0.5 * 256 / np.tan(0.5 * fovx / 180.0 * np.pi)
    fy = 0.5 * 256 / np.tan(0.5 * fovy / 180.0 * np.pi)
    cx = (256 - 1) / 2.0
    cy = (256 - 1) / 2.0
    intrinsics = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]], dtype=np.float32)
    return torch.from_numpy(intrinsics).unsqueeze(0).to(device)


def tensor_to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().clamp(0, 1)
    if img.dim() == 4:
        img = img.squeeze(0)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def get_pano_rgb(output) -> torch.Tensor:
    if hasattr(output.str_output, "sr_image"):
        return output.str_output.sr_image
    return output.str_output.image_raw_compo


def get_per_rgb(output) -> torch.Tensor:
    if hasattr(output.per_output, "sr_image"):
        return output.per_output.sr_image
    return output.per_output.image_raw_compo


# ---------------------------------------------------------------------------
# Mesh export
# ---------------------------------------------------------------------------

def save_obj(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, filepath: str):
    vertices = vertices @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    faces = faces[:, [2, 1, 0]]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_triangle_mesh(filepath, mesh, write_vertex_normals=False)


# ---------------------------------------------------------------------------
# 3D mesh orbit rendering (CPU rasterizer, no OpenGL needed)
# ---------------------------------------------------------------------------

def _rasterize_mesh_frame(
    verts_normalized: np.ndarray,
    faces: np.ndarray,
    face_colors_base: np.ndarray,
    face_normals: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    frame_size: int = 256,
) -> np.ndarray:
    """Rasterize a mesh frame using perspective projection + painter's algorithm.

    Uses OpenCV fillConvexPoly for fast triangle filling with back-face culling.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Camera on a sphere looking at origin
    cam_dist = 1.8
    eye = np.array([
        cam_dist * np.cos(el) * np.sin(az),
        cam_dist * np.cos(el) * np.cos(az),
        cam_dist * np.sin(el),
    ])

    # Look-at matrix
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, np.array([0, 0, 1]))
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([0, 1, 0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # View matrix (world → camera)
    view_rot = np.array([right, up, -forward])
    view_trans = -view_rot @ eye

    # Transform vertices to camera space
    v_cam = (view_rot @ verts_normalized.T).T + view_trans

    # Perspective projection
    fov_half_tan = np.tan(np.radians(22.5))  # 45° FOV
    inv_z = 1.0 / (-v_cam[:, 2] + 1e-8)
    proj_x = v_cam[:, 0] * inv_z / fov_half_tan
    proj_y = v_cam[:, 1] * inv_z / fov_half_tan

    # To pixel coordinates
    half = frame_size * 0.5
    px = (proj_x * half + half).astype(np.int32)
    py = (half - proj_y * half).astype(np.int32)
    depth = -v_cam[:, 2]

    # Directional lighting
    light_dir = np.array([0.3, 0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    shade = np.clip(np.abs(np.sum(face_normals * light_dir, axis=1)), 0.3, 1.0)
    shaded_colors = (face_colors_base * shade[:, None]).astype(np.uint8)

    # Depth sort (painter's algorithm: draw back-to-front)
    face_depths = depth[faces].mean(axis=1)
    sort_order = np.argsort(-face_depths)

    # Build pixel coordinates for all faces
    face_px = np.stack([px[faces], py[faces]], axis=-1)
    sorted_px = face_px[sort_order]
    sorted_colors = shaded_colors[sort_order]

    # Rasterize with OpenCV
    img = np.ones((frame_size, frame_size, 3), dtype=np.uint8) * 255
    for idx in range(len(sort_order)):
        pts = sorted_px[idx]
        color = (int(sorted_colors[idx][2]), int(sorted_colors[idx][1]), int(sorted_colors[idx][0]))
        cv2.fillConvexPoly(img, pts, color)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def render_mesh_orbit_frames(
    mesh_path: str,
    num_frames: int,
    frame_size: int = 256,
) -> List[np.ndarray]:
    """Render a top-side circular orbit around the mesh.

    Uses a fast CPU rasterizer (OpenCV fillConvexPoly + back-face culling)
    that handles the full mesh without subsampling.

    The mesh from Sat3DGen has:
      - X, Z: horizontal ground plane
      - Y: vertical height axis (typically negative)
    We swap Y↔Z so height points "up" for the orbit camera.
    """    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geometries) if geometries else None
    if mesh is None or len(mesh.faces) == 0:
        return [np.ones((frame_size, frame_size, 3), dtype=np.uint8) * 200] * num_frames

    verts = np.array(mesh.vertices, dtype=np.float64)
    faces_arr = np.array(mesh.faces)

    # Vertex colors (stored as RGB in OBJ)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vert_colors = mesh.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
    else:
        vert_colors = np.full((len(verts), 3), 0.6)

    # Swap axes: (X, Y, Z) → (X, Z, Y) so height is "up"
    # Y in mesh is negative (ground below origin), after swap Z = Y stays negative → ground below
    verts_plot = verts.copy()
    verts_plot[:, [1, 2]] = verts[:, [2, 1]]

    # Center and normalize
    center = (verts_plot.max(axis=0) + verts_plot.min(axis=0)) / 2.0
    extent = (verts_plot.max(axis=0) - verts_plot.min(axis=0)).max()
    verts_normalized = (verts_plot - center) / (extent + 1e-8)

    # Precompute face normals and base colors (done once, reused every frame)
    v0 = verts_normalized[faces_arr[:, 0]]
    v1 = verts_normalized[faces_arr[:, 1]]
    v2 = verts_normalized[faces_arr[:, 2]]
    raw_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    face_normals = raw_normals / np.maximum(norms, 1e-8)
    face_colors_base = (vert_colors[faces_arr].mean(axis=1) * 255).astype(np.float32)

    orbit_frames = []
    for frame_idx in range(num_frames):
        azimuth = 360.0 * frame_idx / num_frames
        elevation = 40.0

        frame = _rasterize_mesh_frame(
            verts_normalized, faces_arr, face_colors_base, face_normals,
            azimuth, elevation, frame_size,
        )
        orbit_frames.append(frame)

        if (frame_idx + 1) % 40 == 0 or frame_idx == num_frames - 1:
            print(f"[orbit]    {frame_idx + 1}/{num_frames}")

    return orbit_frames


# ---------------------------------------------------------------------------
# Satellite image with trajectory overlay
# ---------------------------------------------------------------------------

def draw_sat_with_trajectory(
    sat_image_pil: Image.Image,
    pixel_coords: np.ndarray,
    active_index: int,
    target_size: int = 256,
) -> np.ndarray:
    """Draw trajectory on satellite image with active position marker and glow effect. Returns RGB numpy."""
    sat_resized = sat_image_pil.copy().resize((target_size, target_size), Image.BICUBIC)
    sat_np = np.array(sat_resized)

    # Scale pixel coords to target size
    orig_size = sat_image_pil.size[0]
    scale = target_size / orig_size
    scaled_coords = pixel_coords * scale

    # Draw trajectory path (thin white outline + colored line)
    for i in range(len(scaled_coords) - 1):
        pt1 = tuple(np.round(scaled_coords[i]).astype(int))
        pt2 = tuple(np.round(scaled_coords[i + 1]).astype(int))
        cv2.line(sat_np, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
    for i in range(len(scaled_coords) - 1):
        pt1 = tuple(np.round(scaled_coords[i]).astype(int))
        pt2 = tuple(np.round(scaled_coords[i + 1]).astype(int))
        cv2.line(sat_np, pt1, pt2, (255, 80, 80), 2, cv2.LINE_AA)

    # Draw active position with glow effect
    idx = min(active_index, len(scaled_coords) - 1)
    px, py = int(round(scaled_coords[idx][0])), int(round(scaled_coords[idx][1]))

    # Outer glow (semi-transparent via blending)
    overlay = sat_np.copy()
    cv2.circle(overlay, (px, py), 12, (0, 255, 100), -1, cv2.LINE_AA)
    sat_np = cv2.addWeighted(sat_np, 0.7, overlay, 0.3, 0)

    cv2.circle(sat_np, (px, py), 6, (0, 255, 100), -1, cv2.LINE_AA)
    cv2.circle(sat_np, (px, py), 7, (255, 255, 255), 2, cv2.LINE_AA)

    return sat_np


def add_border(image: np.ndarray, thickness: int = 2, color: tuple = (200, 200, 200)) -> np.ndarray:
    """Add a thin border around an image."""
    bordered = image.copy()
    cv2.rectangle(bordered, (0, 0), (bordered.shape[1] - 1, bordered.shape[0] - 1), color, thickness)
    return bordered


def put_label(canvas: np.ndarray, text: str, x: int, y: int,
              font_scale: float = 0.5, color: tuple = (40, 40, 40)):
    """Put a bold label on the canvas."""
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sat_img_path = Path(args.sat_img_path)

    # --- Resolve output directory (create early so trajectory can be saved here) ---
    output_dir = Path(args.work_dir) / sat_img_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve trajectory path ---
    # Priority: 1) explicit --trajectory_path  2) output_dir/trajectory.csv
    traj_in_output = output_dir / "trajectory.csv"
    if args.trajectory_path is not None:
        trajectory_path = Path(args.trajectory_path)
    else:
        trajectory_path = traj_in_output

    if not trajectory_path.exists():
        print(f"[traj]  Not found, launching interactive tool: {trajectory_path}")
        traj_cmd = [
            sys.executable, "inference/make_trajectory.py",
            "--input_img_path", str(sat_img_path),
            "--work_dir", str(output_dir),
        ]
        subprocess.run(traj_cmd, check=True)
        # make_trajectory.py legacy layout saves to {work_dir}/{stem}/pixels.csv
        # New layout (notebook) saves directly to {work_dir}/trajectory.csv
        legacy_csv = output_dir / sat_img_path.stem / "pixels.csv"
        if traj_in_output.exists():
            trajectory_path = traj_in_output
        elif legacy_csv.exists():
            shutil.move(str(legacy_csv), str(traj_in_output))
            gen_dir = output_dir / sat_img_path.stem
            if gen_dir.exists() and not any(gen_dir.iterdir()):
                gen_dir.rmdir()
            trajectory_path = traj_in_output
        else:
            print(f"[traj]  ❌ trajectory still not found (checked {traj_in_output} / {legacy_csv})")
            return

    # --- Load model ---
    checkpoint_path = resolve_checkpoint_path(args.model_path)
    print(f"[model] Loading from {checkpoint_path}")
    # Skip redundant backbone weight download – from_pretrained will
    # overwrite all parameters from the safetensors file anyway.
    Sat3DGen._skip_backbone_weights = True
    model = Sat3DGen.from_pretrained(checkpoint_path).to(device)
    Sat3DGen._skip_backbone_weights = False
    model.eval()

    patch_size = model.unet_model.patch_size if hasattr(model.unet_model, "patch_size") else 16
    sat_transform = T.Compose([
        T.Resize((patch_size * 16, patch_size * 16), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Load satellite image ---
    sat_img_pil = Image.open(sat_img_path).convert("RGB")
    sat_size = sat_img_pil.size[0]
    sat_input = sat_transform(sat_img_pil).unsqueeze(0).to(device)

    # --- Generate triplane ---
    print("[gen]   Triplane features")
    with torch.no_grad():
        triplane = model.from_sat_to_triplane(sat_input)

    # --- Extract mesh ---
    mesh_path = str(output_dir / "mesh.obj")
    print(f"[mesh]  Extracting (res={args.mesh_resolution})")
    with torch.no_grad():
        vertices, faces, vertex_colors = model.extract_mesh(triplane, mesh_resolution=args.mesh_resolution)
    vertices = vertices[:, [1, 2, 0]]
    save_obj(vertices, faces, vertex_colors, mesh_path)
    print(f"[mesh]  {len(vertices)} verts / {len(faces)} faces → {mesh_path}")

    # --- Read trajectory ---
    positions, pixel_coords = read_trajectory_csv(str(trajectory_path), sat_size)
    num_frames = len(positions)
    print(f"[traj]  {num_frames} positions ← {trajectory_path}")

    # --- Prepare sky conditioning ---
    print(f"[sky]   {args.sky_path}")
    sky_hist = make_histo(args.sky_path, device)
    with torch.no_grad():
        w_sky = model.w_sky_prepare(sky_hist)
        sky_feature_2d = model.w_sky2sky_feature_2D(w_sky, sky_hist)

    # --- Render panorama + perspective along trajectory ---
    print(f"[render] Neural rendering {num_frames} frames ...")
    intrinsics = build_intrinsics(device)
    yaw_values = [0, -90, 90, 180]  # front, left, right, back

    pano_rgb_frames = []  # list of RGB numpy (H, W, 3)
    per_row_frames = []   # list of RGB numpy (H, 4*W, 3) — 4 views in a row

    with torch.no_grad():
        for idx, position in enumerate(positions):
            c2w = position_to_c2w(position, device)
            c2w[:, :3, 3] = c2w[:, :3, 3] * model.position_scale_factor

            # Panorama
            pano_result = model.from_3D_to_results(
                triplane, c2w=c2w, w_sky=w_sky, sky_feature_2D=sky_feature_2d, syn_pano=True,
            )
            pano_rgb = tensor_to_numpy_rgb(get_pano_rgb(pano_result))
            pano_rgb_frames.append(pano_rgb)

            # 4 perspective views
            per_views = []
            for yaw in yaw_values:
                c2w_per = c2w.clone()
                c2w_per[:, :3, :3] = torch.from_numpy(compose_rotmat(0, 0, yaw)).unsqueeze(0).to(device)
                per_result = model.from_3D_to_results(
                    triplane, c2w=c2w_per, w_sky=w_sky, intrinsics=intrinsics,
                    sky_feature_2D=sky_feature_2d, syn_pano=False, syn_per=True,
                )
                per_rgb = tensor_to_numpy_rgb(get_per_rgb(per_result))
                per_views.append(per_rgb)

            # Horizontal row: left | front | right | back(flipped) with gaps
            per_back = cv2.flip(per_views[3], 1)
            per_gap = np.ones((per_views[0].shape[0], 4, 3), dtype=np.uint8) * 255
            per_row = np.concatenate([per_views[1], per_gap, per_views[0], per_gap, per_views[2], per_gap, per_back], axis=1)
            per_row_frames.append(per_row)

            if (idx + 1) % 20 == 0 or idx == num_frames - 1:
                print(f"[render]   {idx + 1}/{num_frames}")

    # --- Render 3D mesh orbit ---
    orbit_num = args.orbit_frames if args.orbit_frames > 0 else num_frames
    print(f"[orbit]  Rendering {orbit_num} mesh orbit frames")
    orbit_frames = render_mesh_orbit_frames(
        mesh_path, orbit_num,
        frame_size=512,
    )

    # --- Compose 2x2 grid video frames ---
    print("[video]  Composing frames")
    tmp_frame_dir = output_dir / "_tmp_frames"
    if tmp_frame_dir.exists():
        shutil.rmtree(tmp_frame_dir)
    tmp_frame_dir.mkdir(parents=True)

    # ---- Layout parameters ----
    cell_size = 256
    label_h = 20      # height reserved for text labels above each row
    row_gap = 24      # vertical gap between rows
    row1_gap = 140    # horizontal gap between sat and orbit in row 1
    border_w = 2      # border thickness around each panel
    border_color = (180, 180, 180)
    bg_color = (245, 245, 245)  # light gray background

    pano_h, pano_w = pano_rgb_frames[0].shape[:2]  # 128 x 512
    per_h, per_w = per_row_frames[0].shape[:2]

    # Row 1: sat + orbit
    row1_content_w = cell_size + row1_gap + cell_size

    # Perspective row: keep original size
    per_display_w = per_w
    per_display_h = per_h

    # Panorama: keep native resolution
    pano_display_h = pano_h
    pano_display_w = pano_w

    total_width = max(row1_content_w, pano_w, per_display_w)
    total_width = total_width + (total_width % 2)  # ensure even for ffmpeg

    # Recalculate perspective to match total_width
    per_display_w = total_width
    per_display_h = int(per_h * per_display_w / per_w)

    # Total height: label + row1 + gap + label + row2 + gap + label + row3
    total_height = (label_h + cell_size) + row_gap + (label_h + pano_display_h) + row_gap + (label_h + per_display_h)
    total_height = total_height + (total_height % 2)  # ensure even for ffmpeg

    for frame_idx in range(num_frames):
        composed = np.ones((total_height, total_width, 3), dtype=np.uint8)
        composed[:] = bg_color

        # ---- Row 1: Satellite + Trajectory | 3D Mesh ----
        y1 = 0
        row1_x_start = (total_width - row1_content_w) // 2

        # Labels
        put_label(composed, "Satellite + Trajectory", row1_x_start, y1 + 14)
        put_label(composed, "3D Mesh (orbit)", row1_x_start + cell_size + row1_gap, y1 + 14)

        # Satellite panel
        sat_frame = draw_sat_with_trajectory(sat_img_pil, pixel_coords, frame_idx, cell_size)
        sat_frame = add_border(sat_frame, border_w, border_color)
        composed[y1 + label_h:y1 + label_h + cell_size,
                 row1_x_start:row1_x_start + cell_size] = sat_frame

        # Orbit panel
        orbit_idx = frame_idx % len(orbit_frames)
        orbit_frame = orbit_frames[orbit_idx]
        if orbit_frame.shape[0] != cell_size or orbit_frame.shape[1] != cell_size:
            orbit_frame = cv2.resize(orbit_frame, (cell_size, cell_size))
        orbit_frame = add_border(orbit_frame, border_w, border_color)
        orbit_x = row1_x_start + cell_size + row1_gap
        composed[y1 + label_h:y1 + label_h + cell_size,
                 orbit_x:orbit_x + cell_size] = orbit_frame

        # ---- Row 2: Panorama ----
        y2 = y1 + label_h + cell_size + row_gap
        pano_x_offset = (total_width - pano_display_w) // 2
        put_label(composed, "Panorama", pano_x_offset, y2 + 14)

        pano_frame = pano_rgb_frames[frame_idx]
        if pano_frame.shape[:2] != (pano_display_h, pano_display_w):
            pano_frame = cv2.resize(pano_frame, (pano_display_w, pano_display_h))
        pano_frame = add_border(pano_frame, border_w, border_color)
        composed[y2 + label_h:y2 + label_h + pano_display_h,
                 pano_x_offset:pano_x_offset + pano_display_w] = pano_frame

        # ---- Row 3: Multi-view Perspective ----
        y3 = y2 + label_h + pano_display_h + row_gap
        put_label(composed, "Multi-view Street-level (Left | Front | Right | Back)", 0, y3 + 14)

        per_resized = cv2.resize(per_row_frames[frame_idx], (per_display_w, per_display_h))
        per_resized = add_border(per_resized, border_w, border_color)
        composed[y3 + label_h:y3 + label_h + per_display_h, 0:per_display_w] = per_resized

        frame_path = tmp_frame_dir / f"{frame_idx:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

    # --- Encode video with ffmpeg ---
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("[video]  ❌ ffmpeg not found, frames saved but no mp4")
        return

    print("[video]  Encoding mp4 ...")
    video_path = str(output_dir / "demo_video.mp4")
    subprocess.run([
        ffmpeg_path, "-y",
        "-framerate", str(args.fps),
        "-i", str(tmp_frame_dir / "%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        video_path,
    ], check=True, capture_output=True)

    # --- Encode sub-videos ---
    def encode_sub_video(frames_list, video_name, fps):
        """Encode a list of RGB numpy frames into an mp4 video."""
        sub_dir = output_dir / "_tmp_sub"
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True)
        for i, frame in enumerate(frames_list):
            cv2.imwrite(str(sub_dir / f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_path = str(output_dir / video_name)
        subprocess.run([
            ffmpeg_path, "-y",
            "-framerate", str(fps),
            "-i", str(sub_dir / "%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            out_path,
        ], check=True, capture_output=True)
        shutil.rmtree(sub_dir)
        return out_path

    traj_frames = [draw_sat_with_trajectory(sat_img_pil, pixel_coords, i, 256) for i in range(num_frames)]
    encode_sub_video(traj_frames, "trajectory_video.mp4", args.fps)
    encode_sub_video(orbit_frames, "mesh_orbit_video.mp4", args.fps)
    encode_sub_video(pano_rgb_frames, "panorama_video.mp4", args.fps)
    encode_sub_video(per_row_frames, "streetview_video.mp4", args.fps)

    # --- Clean up composed temp frames ---
    shutil.rmtree(tmp_frame_dir)

    # --- Copy input satellite image as input_sat.png (preserve original extension if not png) ---
    input_sat_save = output_dir / "input_sat.png"
    # Use PIL to ensure consistent .png format regardless of source extension
    sat_img_pil.save(str(input_sat_save))

    # --- Ensure trajectory CSV is at output_dir/trajectory.csv (avoid SameFileError) ---
    target_traj = output_dir / "trajectory.csv"
    try:
        if Path(trajectory_path).resolve() != target_traj.resolve():
            shutil.copy2(str(trajectory_path), str(target_traj))
    except FileNotFoundError:
        pass

    # Mesh is already saved at output_dir/mesh.obj (not deleted)

    print(f"\n✅ Done. Results → {output_dir}")


if __name__ == "__main__":
    main()
