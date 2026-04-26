#!/bin/bash
# Sat3DGen Demo Inference Script
#
# Usage:
#   bash inference.sh                                              # default demo image
#   bash inference.sh <satellite_image_path>                       # custom image
#   bash inference.sh <satellite_image_path> <gpu_id>              # specify GPU
#   bash inference.sh <satellite_image_path> <gpu_id> <sky_path>   # custom sky reference
#
# Prerequisites:
#   - Activate the conda env that has all dependencies installed
#   - Trajectory CSV is generated automatically if missing:
#       * If a graphical display is available, you can also run:
#           python inference/make_trajectory.py --input_img_path <image>
#       * Otherwise, this script will write a config file and wait for
#           inference/make_trajectory.ipynb to produce trajectory.csv

set -e

# --- Configuration ---
# Model checkpoint: local directory or HuggingFace repo id.
# If the local path does not exist, the script will automatically
# download from HuggingFace (qian43/Sat3DGen).
MODEL_PATH="${SAT3DGEN_MODEL_PATH:-qian43/Sat3DGen}"
WORK_DIR="./results/demo"
FPS=5
MESH_RESOLUTION=256

# --- Input arguments ---
# Usage: bash inference.sh <sat_image> [gpu_id] [sky_path]
SAT_IMG="${1:-demo_images/satellite/sat_demo_1.png}"
GPU_ID="${2:-0}"
SKY_PATH="${3:-demo_images/panorama/default_panorama.jpg}"

echo "=============================================="
echo " Sat3DGen Demo Inference"
echo "=============================================="
echo " Satellite image : ${SAT_IMG}"
echo " Sky reference   : ${SKY_PATH}"
echo " Model path      : ${MODEL_PATH}"
echo " Output dir      : ${WORK_DIR}"
echo " GPU             : ${GPU_ID}"
echo "=============================================="

# --- Pre-step: ensure trajectory CSV exists (interactive notebook fallback) ---
SAT_STEM="$(basename "${SAT_IMG}")"; SAT_STEM="${SAT_STEM%.*}"
# Unified output path: ${WORK_DIR}/${SAT_STEM}/trajectory.csv (no nested same-name folder)
TRAJ_CSV="${WORK_DIR}/${SAT_STEM}/trajectory.csv"

if [ ! -f "${TRAJ_CSV}" ]; then
    # --- Validate input image ---
    if [ ! -f "${SAT_IMG}" ]; then
        echo "[ERROR] Input image not found: ${SAT_IMG}"
        echo "        Current dir: $(pwd)"
        exit 1
    fi

    # --- Write a config file so the notebook can read parameters ---
    # (When the notebook is opened manually in VSCode, its kernel does not inherit
    #  shell environment variables, so we pass parameters via a JSON file.)
    # Absolute paths are used to avoid relative-path resolution failures inside the notebook.
    SAVE_DIR="${WORK_DIR}/${SAT_STEM}"
    mkdir -p "${SAVE_DIR}"
    TRAJ_CONFIG="${WORK_DIR}/.traj_config.json"
    # Use Python for abspath conversion (more robust than `cd && pwd`,
    # works even if the path contains special characters).
    SAT_IMG_ABS="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${SAT_IMG}")"
    SAVE_DIR_ABS="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${SAVE_DIR}")"
    cat > "${TRAJ_CONFIG}" <<EOF
{
  "input_img_path": "${SAT_IMG_ABS}",
  "save_dir": "${SAVE_DIR_ABS}",
  "num_points": 79
}
EOF

    echo ""
    echo "============================================================"
    echo " Trajectory file not found: ${TRAJ_CSV}"
    echo " Will use the Jupyter notebook for interactive drawing."
    echo "============================================================"
    echo ""
    echo "[config] Wrote trajectory config to: ${TRAJ_CONFIG}"
    echo "         input_img_path = ${SAT_IMG_ABS}"
    echo "         save_dir       = ${SAVE_DIR_ABS}"
    echo "         trajectory.csv -> ${SAVE_DIR_ABS}/trajectory.csv"
    echo ""
    echo "Next steps:"
    echo "  1. Open in VSCode: inference/make_trajectory.ipynb"
    echo "  2. Run all cells (first time: pip install ipympl)"
    echo "  3. Click and drag on the satellite image to draw a trajectory"
    echo "  4. Run the last cell to save trajectory.csv"
    echo ""
    echo "Waiting for ${TRAJ_CSV} to appear ... (Ctrl+C to abort)"
    echo ""

    # Poll until the trajectory file exists
    WAITED=0
    while [ ! -f "${TRAJ_CSV}" ]; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "   ... waited ${WAITED}s, still waiting for trajectory.csv"
        fi
    done

    echo ""
    echo "[OK] Trajectory file detected, continuing with inference."
    echo ""
fi

# --- Run inference (trajectory is now guaranteed to exist) ---
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference/demo_inference.py \
    --sat_img_path "${SAT_IMG}" \
    --sky_path "${SKY_PATH}" \
    --model_path "${MODEL_PATH}" \
    --work_dir "${WORK_DIR}" \
    --fps ${FPS} \
    --mesh_resolution ${MESH_RESOLUTION}
