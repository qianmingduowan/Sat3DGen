"""Sat3DGen Gradio Demo.

Two-step interactive demo:
  1. Upload a satellite image -> generate and visualize a 3D mesh.
  2. Select a demo image with a pre-generated trajectory -> render panorama + perspective video.
"""

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as T
import trimesh
from PIL import Image

from source.generator import Sat3DGen
from source.rendering.transform_perspective import compose_rotmat

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL: Optional[Sat3DGen] = None
PATCH_SIZE: int = 16
SAT_TRANSFORM = None
RESULTS_DIR = Path("./results/gradio_demo")
TRAJECTORY_PREVIEW_SIZE = 256
DEFAULT_SKY_FILENAMES = (
    "default_panorama.jpg",
    "default_panorama.png",
    "default_panorama.jpeg",
    "default_demo_panorama.jpg",
    "default_demo_panorama.png",
    "default_demo_panorama.jpeg",
    "default_sky.jpg",
    "default_sky.png",
    "default_sky.jpeg",
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


HUGGINGFACE_REPO = "qian43/Sat3DGen"

def load_model(checkpoint_path: str = "checkpoints"):
    """Load the Sat3DGen model (singleton).

    Resolution order:
      1. Local *checkpoint_path* directory (if it contains model files).
      2. HuggingFace Hub repo ``qian43/Sat3DGen``.

    When loading from a full checkpoint (local or Hub), the backbone
    weights are already included in the safetensors file, so the
    standalone DINOv3 download is skipped automatically.
    """
    global MODEL, PATCH_SIZE, SAT_TRANSFORM

    if MODEL is not None:
        return

    model_path: str | None = None
    checkpoint_path_obj = Path(checkpoint_path)
    if (checkpoint_path_obj / "config.json").exists():
        model_path = str(checkpoint_path_obj)
    elif (checkpoint_path_obj / "vqmodel_ema").exists():
        model_path = str(checkpoint_path_obj / "vqmodel_ema")
    elif (checkpoint_path_obj / "vqmodel").exists():
        model_path = str(checkpoint_path_obj / "vqmodel")

    if model_path is None:
        model_path = HUGGINGFACE_REPO
        print(f"Local checkpoint not found at '{checkpoint_path}', loading from HuggingFace: {HUGGINGFACE_REPO}")

    # Skip redundant backbone weight download – from_pretrained will
    # overwrite all parameters from the safetensors file anyway.
    Sat3DGen._skip_backbone_weights = True
    print(f"Loading model from {model_path} ...")
    MODEL = Sat3DGen.from_pretrained(model_path).to(DEVICE)
    Sat3DGen._skip_backbone_weights = False
    MODEL.eval()
    PATCH_SIZE = MODEL.unet_model.patch_size if hasattr(MODEL.unet_model, "patch_size") else 16
    SAT_TRANSFORM = T.Compose([
        T.Resize((PATCH_SIZE * 16, PATCH_SIZE * 16), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Utility helpers (adapted from single_image_inference.py)
# ---------------------------------------------------------------------------

def save_obj(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, filepath: str):
    vertices = vertices @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    faces = faces[:, [2, 1, 0]]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_triangle_mesh(filepath, mesh, write_vertex_normals=False)


def position_to_c2w(position: Tuple[float, float, float]) -> torch.Tensor:
    rotation = compose_rotmat(0, 0, 0)
    pos = np.array(position, dtype=np.float32)
    pos[0] *= -1
    pos = pos[[1, 0, 2]]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(rotation, dtype=np.float32)
    c2w[:3, 3] = pos
    return torch.from_numpy(c2w).unsqueeze(0).to(DEVICE)


def build_intrinsics() -> torch.Tensor:
    fovx, fovy = 120, 120
    fx = 0.5 * 256 / np.tan(0.5 * fovx / 180.0 * np.pi)
    fy = 0.5 * 256 / np.tan(0.5 * fovy / 180.0 * np.pi)
    cx = (256 - 1) / 2.0
    cy = (256 - 1) / 2.0
    intrinsics = np.array([[fx / 2, 0, cx / 2], [0, fy / 2, cy / 2], [0, 0, 1]], dtype=np.float32)
    return torch.from_numpy(intrinsics).unsqueeze(0).to(DEVICE)


def tensor_to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [1, C, H, W] or [C, H, W] tensor in [0, 1] to a uint8 RGB numpy array."""
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


def make_histo(grd_img_path: str) -> torch.Tensor:
    grd_img = Image.open(grd_img_path).convert("RGB").resize((512, 128))
    grd_img = T.ToTensor()(grd_img).unsqueeze(0).float().to(DEVICE)

    # Derive the sky-mask path by replacing only the parent directory name,
    # keeping the filename intact (just switching extension to .png).
    grd_path = Path(grd_img_path)
    parent_name = grd_path.parent.name
    if parent_name in ("streetview", "panorama"):
        mask_dir = grd_path.parent.parent / "pano_sky_mask"
        mask_img_path = str(mask_dir / grd_path.with_suffix(".png").name)
    else:
        raise ValueError(f"Cannot infer sky-mask path from {grd_img_path}")

    mask_img = Image.open(mask_img_path).convert("L").resize((512, 128), Image.NEAREST)
    mask_img = T.ToTensor()(mask_img).unsqueeze(0).float().to(DEVICE)

    sky_image = (grd_img * mask_img).mul(2).sub(1)
    sky_image = sky_image.detach().cpu().numpy()

    from source.sky_histogram import compute_sky_histogram

    histo_sky = torch.from_numpy(
        compute_sky_histogram(sky_image[0], hist_range=(-1, 1))
    ).unsqueeze(0).float().to(DEVICE)
    return histo_sky


def read_trajectory_from_csv(csv_path: str, sat_image_size: int) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
    """Read a pre-generated trajectory .csv file (format: w,h,angle).

    Returns:
        positions: list of (x_norm, y_norm, z) in [-1, 1] range for rendering
        pixel_coords: Nx2 array of pixel coordinates for visualization
    """
    half = sat_image_size / 2
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


def draw_trajectory_on_satellite(
    sat_image_pil: Image.Image,
    pixel_coords: np.ndarray,
    active_index: Optional[int] = None,
) -> np.ndarray:
    """Draw trajectory on satellite image with glow effect (matching demo_inference style)."""
    sat_frame = np.array(sat_image_pil.convert("RGB"))

    if len(pixel_coords) >= 2:
        # White outline pass (thicker, drawn first)
        for idx in range(len(pixel_coords) - 1):
            pt1 = tuple(np.round(pixel_coords[idx]).astype(int))
            pt2 = tuple(np.round(pixel_coords[idx + 1]).astype(int))
            cv2.line(sat_frame, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
        # Colored line pass (thinner, on top)
        for idx in range(len(pixel_coords) - 1):
            pt1 = tuple(np.round(pixel_coords[idx]).astype(int))
            pt2 = tuple(np.round(pixel_coords[idx + 1]).astype(int))
            cv2.line(sat_frame, pt1, pt2, (255, 80, 80), 2, cv2.LINE_AA)

    if active_index is not None and len(pixel_coords) > 0:
        coord = pixel_coords[min(active_index, len(pixel_coords) - 1)]
        px, py = int(round(coord[0])), int(round(coord[1]))
        # Outer glow via alpha blending
        overlay = sat_frame.copy()
        cv2.circle(overlay, (px, py), 12, (0, 255, 100), -1, cv2.LINE_AA)
        sat_frame = cv2.addWeighted(sat_frame, 0.7, overlay, 0.3, 0)
        # Solid inner circle + white ring
        cv2.circle(sat_frame, (px, py), 6, (0, 255, 100), -1, cv2.LINE_AA)
        cv2.circle(sat_frame, (px, py), 7, (255, 255, 255), 2, cv2.LINE_AA)

    return sat_frame


def build_trajectory_preview(sat_image_pil: Image.Image, pixel_coords: np.ndarray) -> Image.Image:
    sat_frame = draw_trajectory_on_satellite(sat_image_pil, pixel_coords)
    preview = cv2.resize(
        sat_frame,
        (TRAJECTORY_PREVIEW_SIZE, TRAJECTORY_PREVIEW_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    return Image.fromarray(preview)


def resolve_demo_sky_pairs(demo_dir: Path) -> Tuple[List[Tuple[Path, Path]], Optional[Path]]:
    pano_dir = demo_dir / "panorama"
    mask_dir = demo_dir / "pano_sky_mask"
    if not pano_dir.exists() or not mask_dir.exists():
        return [], None

    mask_lookup = {mask_path.stem: mask_path for mask_path in sorted(mask_dir.glob("*.png"))}
    sky_pairs: List[Tuple[Path, Path]] = []
    for pano_path in sorted(pano_dir.glob("*")):
        if pano_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_path = mask_lookup.get(pano_path.stem)
        if mask_path is not None:
            sky_pairs.append((pano_path, mask_path))

    if not sky_pairs:
        return [], None

    default_idx = 0
    for idx, (pano_path, _) in enumerate(sky_pairs):
        pano_name_lower = pano_path.name.lower()
        pano_stem_lower = pano_path.stem.lower()
        if pano_name_lower in DEFAULT_SKY_FILENAMES or "default" in pano_stem_lower:
            default_idx = idx
            break

    ordered_pairs = [sky_pairs[default_idx], *sky_pairs[:default_idx], *sky_pairs[default_idx + 1 :]]
    return ordered_pairs, ordered_pairs[0][0]


# ---------------------------------------------------------------------------
# Step 1: Satellite Image → 3D Mesh
# ---------------------------------------------------------------------------

def generate_mesh(sat_image_pil: Image.Image, mesh_resolution: int = 256, progress=gr.Progress()):
    """Generate a 3D mesh from a satellite image."""
    if sat_image_pil is None:
        raise gr.Error("Please upload a satellite image first.")

    print("[generate_mesh] >>> Start")
    load_model()
    print("[generate_mesh] Model loaded")

    progress(0.1, desc="Preprocessing satellite image...")
    print("[generate_mesh] Preprocessing satellite image...")
    sat_input = SAT_TRANSFORM(sat_image_pil.convert("RGB")).unsqueeze(0).to(DEVICE)

    progress(0.3, desc="Generating triplane features...")
    print("[generate_mesh] Generating triplane features...")
    with torch.no_grad():
        triplane = MODEL.from_sat_to_triplane(sat_input)
    print("[generate_mesh] Triplane generated successfully")

    progress(0.5, desc="Extracting 3D mesh (this may take a moment)...")
    print(f"[generate_mesh] Extracting 3D mesh (resolution={mesh_resolution})...")
    with torch.no_grad():
        vertices, faces, vertex_colors = MODEL.extract_mesh(triplane, mesh_resolution=mesh_resolution)
    print(f"[generate_mesh] Mesh extracted: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    vertices = vertices[:, [1, 2, 0]]

    # Save mesh
    mesh_path = str(RESULTS_DIR / "mesh.obj")
    save_obj(vertices, faces, vertex_colors, mesh_path)
    print(f"[generate_mesh] OBJ saved to {mesh_path}")

    # Also save triplane to state for Step 2
    state = {"triplane": triplane, "sat_image": sat_image_pil}

    progress(0.9, desc="Preparing 3D visualization...")
    print("[generate_mesh] Converting OBJ → GLB for 3D preview...")

    # Create a glb file for Gradio's Model3D component.
    # Use a tempfile so Gradio can reliably serve it via its file cache.
    import tempfile, shutil
    glb_path_local = str(RESULTS_DIR / "mesh.glb")
    mesh_trimesh = trimesh.load(mesh_path, process=False)
    # Ensure we have a single Trimesh (not a Scene) with vertex normals,
    # otherwise Chrome's WebGL renderer shows a blank canvas.
    if isinstance(mesh_trimesh, trimesh.Scene):
        geometries = list(mesh_trimesh.geometry.values())
        if geometries:
            mesh_trimesh = trimesh.util.concatenate(geometries)
        else:
            raise gr.Error("Failed to load mesh geometry.")
    if not hasattr(mesh_trimesh, 'vertex_normals') or mesh_trimesh.vertex_normals is None or len(mesh_trimesh.vertex_normals) == 0:
        mesh_trimesh.vertex_normals  # triggers auto-computation
    print(f"[generate_mesh] Mesh has {len(mesh_trimesh.vertices)} verts, {len(mesh_trimesh.faces)} faces, normals: {mesh_trimesh.vertex_normals.shape}")
    mesh_trimesh.export(glb_path_local, file_type="glb")
    print(f"[generate_mesh] GLB saved to {glb_path_local} ({os.path.getsize(glb_path_local)} bytes)")

    tmp_glb = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    shutil.copy2(glb_path_local, tmp_glb.name)
    tmp_glb.close()
    print(f"[generate_mesh] GLB copied to temp file: {tmp_glb.name}")

    progress(1.0, desc="Done!")
    print("[generate_mesh] <<< 3D mesh generated successfully!")
    return tmp_glb.name, mesh_path, state


def download_mesh(mesh_path: str):
    """Return the mesh file for download."""
    if mesh_path and os.path.exists(mesh_path):
        return mesh_path
    return None


# ---------------------------------------------------------------------------
# Step 2: Trajectory → Panorama + Perspective Video
# ---------------------------------------------------------------------------

def render_trajectory_video(
    sat_image_pil: Image.Image,
    trajectory_csv_path: str,
    sky_path: str,
    progress=gr.Progress(),
):
    """Render panorama and perspective views along a pre-generated trajectory.

    Layout per frame:
      Top row:    satellite image (with camera marker)  |  panorama RGB
      Bottom row: 4 perspective views in a horizontal row (left, front, right, back)
    """
    print("[render_trajectory_video] >>> Start")
    load_model()

    sat_size = sat_image_pil.size[0]
    positions, pixel_coords = read_trajectory_from_csv(trajectory_csv_path, sat_size)
    if len(positions) == 0:
        raise gr.Error(f"Trajectory file is empty: {trajectory_csv_path}")
    print(f"[render_trajectory_video] Loaded {len(positions)} positions from {trajectory_csv_path}")

    progress(0.1, desc="Extracting triplane features...")
    sat_tensor = SAT_TRANSFORM(sat_image_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        triplane = MODEL.from_sat_to_triplane(sat_tensor)

    progress(0.2, desc="Preparing sky condition...")
    sky_hist = make_histo(sky_path)
    with torch.no_grad():
        w_sky = MODEL.w_sky_prepare(sky_hist)
        sky_feature_2d = MODEL.w_sky2sky_feature_2D(w_sky, sky_hist)

    progress(0.25, desc="Rendering views along trajectory...")
    intrinsics = build_intrinsics()
    yaw_values = [0, -90, 90, 180]

    video_dir = RESULTS_DIR / "video_frames"
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    total_positions = len(positions)
    for idx, position in enumerate(positions):
        progress(0.25 + 0.6 * idx / total_positions, desc=f"Rendering frame {idx + 1}/{total_positions}...")
        if idx % 10 == 0 or idx == total_positions - 1:
            print(f"[render_trajectory_video] Rendering frame {idx + 1}/{total_positions}...")

        c2w = position_to_c2w(position)
        c2w[:, :3, 3] = c2w[:, :3, 3] * MODEL.position_scale_factor

        with torch.no_grad():
            pano_result = MODEL.from_3D_to_results(
                triplane,
                c2w=c2w,
                w_sky=w_sky,
                sky_feature_2D=sky_feature_2d,
                syn_pano=True,
            )
            pano_rgb = tensor_to_numpy_rgb(get_pano_rgb(pano_result))

            per_views = []
            for yaw in yaw_values:
                c2w_per = c2w.clone()
                c2w_per[:, :3, :3] = torch.from_numpy(compose_rotmat(0, 0, yaw)).unsqueeze(0).to(DEVICE)
                per_result = MODEL.from_3D_to_results(
                    triplane,
                    c2w=c2w_per,
                    w_sky=w_sky,
                    intrinsics=intrinsics,
                    sky_feature_2D=sky_feature_2d,
                    syn_pano=False,
                    syn_per=True,
                )
                per_rgb = tensor_to_numpy_rgb(get_per_rgb(per_result))
                per_views.append(per_rgb)

        # --- Satellite image with camera position marker ---
        sat_frame = draw_trajectory_on_satellite(sat_image_pil, pixel_coords, active_index=idx)

        # --- Compose frame ---
        # Top row: satellite (square) | panorama RGB
        pano_h, pano_w = pano_rgb.shape[:2]
        sat_resized = cv2.resize(sat_frame, (pano_h, pano_h))
        top_row = np.concatenate([sat_resized, pano_rgb], axis=1)

        # Bottom row: 4 perspective views in a horizontal row (left, front, right, back)
        # Flip back view for consistency
        per_back = cv2.flip(per_views[3], 1)
        per_row = np.concatenate([per_views[1], per_views[0], per_views[2], per_back], axis=1)

        # Resize bottom row to match top row width
        top_width = top_row.shape[1]
        per_row_h = int(per_row.shape[0] * top_width / per_row.shape[1])
        per_row_resized = cv2.resize(per_row, (top_width, per_row_h))

        composed = np.concatenate([top_row, per_row_resized], axis=0)

        frame_path = video_dir / f"{idx:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

    progress(0.9, desc="Encoding video...")
    print("[render_trajectory_video] All frames rendered, encoding video with ffmpeg...")
    video_path = str(RESULTS_DIR / "trajectory_video.mp4")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise gr.Error("ffmpeg not found. Please install ffmpeg to generate videos.")

    subprocess.run([
        ffmpeg_path, "-y", "-framerate", "5",
        "-i", str(video_dir / "%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        video_path,
    ], check=True, capture_output=True)

    print(f"[render_trajectory_video] Video saved to {video_path}")
    progress(1.0, desc="Done!")
    return video_path



# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo():

    # Find sample images from demo directory
    demo_dir = Path(__file__).resolve().parent / "demo_images"
    sample_sat_images = sorted((demo_dir / "satellite").glob("*.png")) if (demo_dir / "satellite").exists() else []
    sample_sat_images_with_csv = [p for p in sample_sat_images if p.with_suffix(".csv").exists()]
    sample_sky_pairs, default_sky_path = resolve_demo_sky_pairs(demo_dir)

    with gr.Blocks(title="Sat3DGen Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            ## Sat3DGen: 3D Scene Generation from Satellite Imagery

            Upload a satellite image to **generate a 3D mesh** or **render a walkthrough video**.

            📌 **Input requirements:** The satellite image should be at **zoom level 20**
            (same as the [VIGOR](https://github.com/Jeff-Zilence/VIGOR) dataset), then will be resized to the input size.
            You can download satellite tiles at this zoom level from any map tile API (e.g. Google Maps, Bing Maps, Mapbox).
            """
        )

        # Shared state
        inference_state = gr.State(value=None)
        mesh_file_path = gr.State(value=None)

        # ---- 3D Mesh Generation ----
        with gr.Tab("3D Mesh Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    sat_input = gr.Image(
                        label="Upload Satellite Image",
                        type="pil",
                        height=400,
                    )
                    mesh_resolution_slider = gr.Slider(
                        minimum=128, maximum=512, value=128, step=64,
                        label="Mesh Resolution (voxel size)",
                    )
                    generate_button = gr.Button("🚀 Generate 3D Mesh", variant="primary", size="lg")

                with gr.Column(scale=2):
                    mesh_viewer = gr.Model3D(label="3D Mesh Preview", height=500)
                    download_button = gr.DownloadButton("💾 Download Mesh (.obj)", variant="secondary")

            if sample_sat_images:
                gr.Markdown("### Sample Images")
                gr.Examples(
                    examples=[[str(p)] for p in sample_sat_images],
                    inputs=[sat_input],
                    label="Click to load a sample satellite image",
                    examples_per_page=len(sample_sat_images),
                )

            gr.Markdown(
                "⚠️ **Note:** The 3D mesh preview may show slight color distortion. "
                "The cause is currently under investigation."
            )

            generate_button.click(
                fn=generate_mesh,
                inputs=[sat_input, mesh_resolution_slider],
                outputs=[mesh_viewer, mesh_file_path, inference_state],
            )
            mesh_file_path.change(
                fn=download_mesh,
                inputs=[mesh_file_path],
                outputs=[download_button],
            )
        # ---- Video Rendering ----
        with gr.Tab("Video Rendering"):
            # Hidden state to track the resolved trajectory .csv path
            trajectory_csv_state = gr.State(value=None)
            sky_path_state = gr.State(value=str(default_sky_path) if default_sky_path is not None else None)

            def load_sat_from_gallery(evt: gr.SelectData):
                """Load selected satellite image and check for a same-name trajectory .csv."""
                if evt.index is None or evt.index >= len(sample_sat_images_with_csv):
                    return None, None, "No image selected.", None
                sat_path = sample_sat_images_with_csv[evt.index]
                sat_pil = Image.open(str(sat_path))
                csv_path = sat_path.with_suffix(".csv")
                if csv_path.exists():
                    status_msg = f"✅ Trajectory found: `{csv_path.name}`"
                    _, pixel_coords = read_trajectory_from_csv(str(csv_path), sat_pil.size[0])
                    preview = build_trajectory_preview(sat_pil, pixel_coords)
                    return sat_pil, str(csv_path), status_msg, preview
                status_msg = (
                    f"⚠️ No trajectory file found. "
                    f"Please pre-generate a trajectory and save it as "
                    f"`{csv_path.name}` in `{sat_path.parent}/` using:\n\n"
                    f"```\npython inference/make_trajectory.py "
                    f"--input_img_path {sat_path} --save_same_name\n```"
                )
                return sat_pil, None, status_msg, None

            def on_sat_upload(sat_image_pil):
                """When user uploads a custom satellite image, no same-name trajectory CSV is available."""
                if sat_image_pil is None:
                    return None, "No image uploaded.", None
                return None, (
                    "⚠️ For uploaded images, you need a **trajectory .csv** file with the same name "
                    "as your satellite image (e.g. `my_image.csv` for `my_image.png`).\n\n"
                    "You can generate one interactively using either:\n\n"
                    "- **Jupyter Notebook** (recommended): `inference/make_trajectory.ipynb`\n"
                    "- **Command line**: "
                    "`python inference/make_trajectory.py --input_img_path <your_image_path> --save_same_name`\n\n"
                    "If you used the command line **without** `--save_same_name`, "
                    "the CSV is saved under `results/<image_name>/pixels.csv`. "
                    "You will need to **copy** it next to your satellite image with the same base name "
                    "(e.g. copy to `demo_images/satellite/my_image.csv`)."
                ), None

            def load_sky_from_gallery(evt: gr.SelectData):
                """Select one demo panorama street image. The first entry is the default."""
                if not sample_sky_pairs:
                    return None, None, "No demo panorama street image is available."
                if evt.index is None or evt.index >= len(sample_sky_pairs):
                    sky_path = default_sky_path
                else:
                    sky_path = sample_sky_pairs[evt.index][0]
                default_suffix = " (Default)" if sky_path == default_sky_path else ""
                status_msg = (
                    f"Selected demo panorama: `{sky_path.name}`{default_suffix}\n\n"
                    f"If you do not choose another one, this image will be used."
                )
                return str(sky_path), str(sky_path), status_msg

            def render_video_from_state(sat_image, csv_path, sky_path, progress=gr.Progress()):
                """Render video using the pre-generated trajectory CSV."""
                if sat_image is None:
                    raise gr.Error("Please select or upload a satellite image first.")
                if csv_path is None or not Path(csv_path).exists():
                    raise gr.Error(
                        "No trajectory CSV found. Please pre-generate a trajectory using: "
                        "python inference/make_trajectory.py --input_img_path <image> --save_same_name"
                    )
                resolved_sky_path = sky_path or (str(default_sky_path) if default_sky_path is not None else None)
                if resolved_sky_path is None or not Path(resolved_sky_path).exists():
                    raise gr.Error("No valid demo panorama is available. Please add one under demo_images/panorama.")
                return render_trajectory_video(sat_image, csv_path, resolved_sky_path, progress)

            # ===== Main layout =====
            with gr.Row(equal_height=False):
                # Left column: satellite image selection
                with gr.Column(scale=1):
                    sat_input_video = gr.Image(
                        label="Upload Satellite Image",
                        type="pil",
                        height=300,
                    )
                    trajectory_status = gr.Markdown(value="Select a demo image or upload your own.")
                    selected_sky_preview = gr.Image(
                        label="Selected Demo Panorama",
                        value=str(default_sky_path) if default_sky_path is not None else None,
                        height=180,
                    )
                    default_sky_message = "No demo panorama street image is available."
                    if default_sky_path is not None:
                        default_sky_message = (
                            f"Default demo panorama: `{default_sky_path.name}`\n\n"
                            "If you do not select another demo panorama, this one will be used."
                        )
                    sky_status = gr.Markdown(value=default_sky_message)
                    render_button = gr.Button("🎬 Render Video", variant="primary", size="lg")

                # Middle column: trajectory preview
                with gr.Column(scale=1):
                    trajectory_preview = gr.Image(label="Trajectory Preview", height=300)

                # Right column: video output
                with gr.Column(scale=2):
                    video_output = gr.Video(label="Rendered Video", height=500)

            # ===== Sample Satellite Images Gallery (only those with a trajectory CSV) =====
            if sample_sat_images_with_csv:
                gr.Markdown("### 🛰️ Sample Satellite Images — click to load")
                sat_gallery = gr.Gallery(
                    value=[str(p) for p in sample_sat_images_with_csv],
                    label="Sample Satellite Images (with trajectory)",
                    columns=5,
                    rows=1,
                    height=120,
                    object_fit="cover",
                    allow_preview=False,
                )
                sat_gallery.select(
                    fn=load_sat_from_gallery,
                    inputs=None,
                    outputs=[sat_input_video, trajectory_csv_state, trajectory_status, trajectory_preview],
                )

            if sample_sky_pairs:
                gr.Markdown(
                    "### 🌤️ Demo Panorama Street Images — the first one is the default\n\n"
                    "The panorama image and its corresponding sky mask are used to extract a "
                    "**sky region color histogram**, which serves as a **lighting condition hint** "
                    "during street-view rendering. This only affects the appearance (illumination/color tone) "
                    "of the rendered views — it does **not** alter the underlying 3D NeRF geometry."
                )
                sky_gallery = gr.Gallery(
                    value=[
                        (
                            str(pano_path),
                            f"{pano_path.name} (Default)" if pano_path == default_sky_path else pano_path.name,
                        )
                        for pano_path, _ in sample_sky_pairs
                    ],
                    label="Demo Panorama Street Images",
                    columns=5,
                    rows=1,
                    height=120,
                    object_fit="cover",
                    allow_preview=False,
                )
                sky_gallery.select(
                    fn=load_sky_from_gallery,
                    inputs=None,
                    outputs=[sky_path_state, selected_sky_preview, sky_status],
                )

            # When user uploads a custom image
            sat_input_video.upload(
                fn=on_sat_upload,
                inputs=[sat_input_video],
                outputs=[trajectory_csv_state, trajectory_status, trajectory_preview],
            )

            render_button.click(
                fn=render_video_from_state,
                inputs=[sat_input_video, trajectory_csv_state, sky_path_state],
                outputs=[video_output],
            )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        allowed_paths=[
            str(Path(__file__).resolve().parent / "demo_images"),
            str(Path(__file__).resolve().parent / "results"),
        ],
    )
