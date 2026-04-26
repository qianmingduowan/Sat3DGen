#!/usr/bin/env bash
# =============================================================================
# run_inference_video.sh
#
# End-to-end pipeline: make_trajectory → single_image_inference → combined video
#
# Usage:
#   bash run_inference_video.sh \
#       --sat_img  <satellite_image_path> \
#       --csv      <trajectory_csv_path>  \
#       [--sky     <panorama_image_path>] \
#       [--ckpt    <checkpoint_dir>]      \
#       [--workdir <output_root>]         \
#       [--gpu     <gpu_id>]
#
# Example (with existing trajectory CSV):
#   bash run_inference_video.sh \
#       --sat_img /path/to/satellite.png \
#       --csv     /path/to/pixels.csv
#
# Example (draw trajectory interactively first):
#   bash run_inference_video.sh \
#       --sat_img /path/to/satellite.png
# =============================================================================
set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
WORK_DIR="${SCRIPT_DIR}/results"
SKY_PATH="${SCRIPT_DIR}/../data/vigor/Chicago/panorama/_DQh9bpzvE0UpcPPqLu_DQ,41.863472,-87.657822,.jpg"
GPU_ID=0
SAT_IMG=""
CSV_PATH=""
NUM_POINTS=79

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sat_img)  SAT_IMG="$2";   shift 2 ;;
        --csv)      CSV_PATH="$2";  shift 2 ;;
        --sky)      SKY_PATH="$2";  shift 2 ;;
        --ckpt)     CKPT_DIR="$2";  shift 2 ;;
        --workdir)  WORK_DIR="$2";  shift 2 ;;
        --gpu)      GPU_ID="$2";    shift 2 ;;
        --num_points) NUM_POINTS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${SAT_IMG}" ]]; then
    echo "Error: --sat_img is required."
    echo "Usage: bash $0 --sat_img <path> [--csv <path>] [--sky <path>]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Derive a clean output name from the satellite image
SAT_BASENAME="$(basename "${SAT_IMG}" | sed 's/\.[^.]*$//')"
TRAJ_DIR="${WORK_DIR}/${SAT_BASENAME}"

echo "=============================================="
echo " Sat3DGen Inference Pipeline"
echo "=============================================="
echo " Satellite image : ${SAT_IMG}"
echo " Checkpoint      : ${CKPT_DIR}"
echo " Sky panorama    : ${SKY_PATH}"
echo " Output root     : ${WORK_DIR}"
echo " GPU             : ${GPU_ID}"
echo "=============================================="

# =============================================================================
# Step 1: Make trajectory (skip if CSV already provided)
# =============================================================================
if [[ -z "${CSV_PATH}" ]]; then
    echo ""
    echo "[Step 1/3] Drawing trajectory interactively..."
    echo "  → A matplotlib window will open. Left-click and drag to draw a path."
    echo "  → Release the mouse button when done."
    cd "${SCRIPT_DIR}"
    python inference/make_trajectory.py \
        --input_img_path "${SAT_IMG}" \
        --work_dir "${WORK_DIR}" \
        --num_of_point "${NUM_POINTS}"

    CSV_PATH="${TRAJ_DIR}/pixels.csv"
    if [[ ! -f "${CSV_PATH}" ]]; then
        echo "Error: Trajectory CSV not found at ${CSV_PATH}"
        exit 1
    fi
    echo "  ✓ Trajectory saved to ${CSV_PATH}"
else
    echo ""
    echo "[Step 1/3] Using existing trajectory CSV: ${CSV_PATH}"
fi

# =============================================================================
# Step 2: Run single_image_inference with --save_video
# =============================================================================
echo ""
echo "[Step 2/3] Running single-image inference + video rendering..."
cd "${SCRIPT_DIR}"
python inference/single_image_inference.py \
    --sat_img_path "${SAT_IMG}" \
    --model_path "${CKPT_DIR}" \
    --position_path "${CSV_PATH}" \
    --sky_path "${SKY_PATH}" \
    --work_dir "${WORK_DIR}" \
    --save_video \
    --save_sat \
    --save_sky \
    --save_street \
    --visual_direction

# Locate the output directory (resolve_output_dir logic)
CKPT_PARENT="$(basename "$(dirname "${CKPT_DIR}")")"
CKPT_NAME="$(basename "${CKPT_DIR}")"
OUTPUT_DIR="${WORK_DIR}/${SAT_BASENAME}/${CKPT_PARENT}/${CKPT_NAME}"

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    # Fallback: try to find the output directory
    OUTPUT_DIR="$(find "${WORK_DIR}/${SAT_BASENAME}" -name "save_vid" -type d -print -quit 2>/dev/null | xargs dirname 2>/dev/null || true)"
    if [[ -z "${OUTPUT_DIR}" || ! -d "${OUTPUT_DIR}" ]]; then
        echo "Error: Cannot locate inference output directory."
        echo "Expected: ${WORK_DIR}/${SAT_BASENAME}/${CKPT_PARENT}/${CKPT_NAME}"
        exit 1
    fi
fi

echo "  ✓ Inference outputs saved to ${OUTPUT_DIR}"

# =============================================================================
# Step 3: Compose a combined video (sat+trajectory | panorama | perspective)
# =============================================================================
echo ""
echo "[Step 3/3] Composing combined video..."

COMBINED_DIR="${OUTPUT_DIR}/combined_frames"
mkdir -p "${COMBINED_DIR}"

python3 - "${OUTPUT_DIR}" "${COMBINED_DIR}" "${SAT_IMG}" "${CSV_PATH}" <<'PYEOF'
import sys
import os
import csv
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw

output_dir = sys.argv[1]
combined_dir = sys.argv[2]
sat_img_path = sys.argv[3]
csv_path = sys.argv[4]

# Load satellite image
sat_img = Image.open(sat_img_path).convert("RGB")
sat_size = sat_img.size[0]
half_pixel = sat_size // 2

# Read trajectory
trajectory = []
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        w, h = float(row["w"]), float(row["h"])
        trajectory.append((w, h))

# Collect frame paths
sat_frames = sorted(glob.glob(os.path.join(output_dir, "save_sat", "*.png")))
street_frames = sorted(glob.glob(os.path.join(output_dir, "save_street_vid", "*.png")))
per_frames = sorted(glob.glob(os.path.join(output_dir, "save_street_vid_per", "*.png")))

num_frames = len(sat_frames)
print(f"  Found {num_frames} frames (sat={len(sat_frames)}, street={len(street_frames)}, per={len(per_frames)})")

if num_frames == 0:
    print("  Error: No frames found!")
    sys.exit(1)

# Target height for all panels
TARGET_H = 512

for idx in range(num_frames):
    # --- Panel 1: Satellite with trajectory overlay ---
    sat_canvas = sat_img.copy()
    draw = ImageDraw.Draw(sat_canvas)

    # Draw full trajectory path
    for i in range(len(trajectory) - 1):
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill="cyan", width=2)

    # Draw current position marker
    if idx < len(trajectory):
        cx, cy = trajectory[idx]
        radius = 5
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill="red", outline="white", width=1,
        )

    sat_np = cv2.cvtColor(np.array(sat_canvas), cv2.COLOR_RGB2BGR)
    # Resize satellite to TARGET_H x TARGET_H
    sat_panel = cv2.resize(sat_np, (TARGET_H, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # --- Panel 2: Panorama + depth (from save_street_vid) ---
    if idx < len(street_frames):
        street_panel = cv2.imread(street_frames[idx])
    else:
        street_panel = np.zeros((TARGET_H, TARGET_H, 3), dtype=np.uint8)

    # Resize street panel to match height
    sh, sw = street_panel.shape[:2]
    new_sw = int(sw * TARGET_H / sh)
    street_panel = cv2.resize(street_panel, (new_sw, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # --- Panel 3: Perspective views (from save_street_vid_per) ---
    per_idx = idx  # perspective frames are saved every 4 original frames
    per_name = f"{idx:03d}per.png"
    per_path = os.path.join(output_dir, "save_street_vid_per", per_name)
    if os.path.exists(per_path):
        per_panel = cv2.imread(per_path)
    else:
        per_panel = np.zeros((TARGET_H, TARGET_H, 3), dtype=np.uint8)

    # Resize perspective panel to match height
    ph, pw = per_panel.shape[:2]
    if ph > 0:
        new_pw = int(pw * TARGET_H / ph)
        per_panel = cv2.resize(per_panel, (new_pw, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # --- Compose all panels side by side ---
    combined = np.concatenate([sat_panel, street_panel, per_panel], axis=1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_y = 30
    cv2.putText(combined, "Satellite + Trajectory", (10, label_y), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Panorama + Depth", (TARGET_H + 10, label_y), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Perspective Views", (TARGET_H + new_sw + 10, label_y), font, 0.7, (255, 255, 255), 2)

    # Frame counter
    cv2.putText(combined, f"Frame {idx+1}/{num_frames}", (10, combined.shape[0] - 15), font, 0.5, (200, 200, 200), 1)

    save_path = os.path.join(combined_dir, f"{idx:04d}.png")
    cv2.imwrite(save_path, combined)

print(f"  ✓ {num_frames} combined frames saved to {combined_dir}")
PYEOF

# Encode combined video with ffmpeg
COMBINED_VIDEO="${OUTPUT_DIR}/combined_video.mp4"
ffmpeg -y -framerate 5 \
    -i "${COMBINED_DIR}/%04d.png" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -pix_fmt yuv420p \
    "${COMBINED_VIDEO}" 2>/dev/null

echo ""
echo "=============================================="
echo " ✅ Pipeline complete!"
echo "=============================================="
echo " Output directory  : ${OUTPUT_DIR}"
echo " Combined video    : ${COMBINED_VIDEO}"
echo " Individual videos : ${OUTPUT_DIR}/vid.mp4"
echo "                     ${OUTPUT_DIR}/vid_str_per.mp4"
echo "=============================================="
