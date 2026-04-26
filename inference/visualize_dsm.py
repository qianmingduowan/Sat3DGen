import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DSM `.npz` files as color-mapped PNG images.")
    parser.add_argument("--input_path", type=str, required=True, help="A DSM `.npz` file or a directory of `.npz` files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory used to save the PNG visualizations.")
    parser.add_argument("--key", type=str, default=None, help="Optional array key inside the `.npz` file.")
    parser.add_argument("--invert", action="store_true", help="Invert the normalized values before visualization.")
    parser.add_argument("--percentile_min", type=float, default=2.0)
    parser.add_argument("--percentile_max", type=float, default=98.0)
    return parser.parse_args()


def load_array(npz_path: Path, key: Optional[str]) -> np.ndarray:
    data = np.load(npz_path)
    if key is not None:
        return data[key]
    if "dsm" in data:
        return data["dsm"]
    if "density" in data:
        return data["density"]
    return data[data.files[0]]


def normalize_to_uint8(array: np.ndarray, percentile_min: float, percentile_max: float, invert: bool) -> np.ndarray:
    valid_mask = np.isfinite(array)
    if not np.any(valid_mask):
        return np.zeros(array.shape, dtype=np.uint8)

    valid_values = array[valid_mask]
    vmin = np.percentile(valid_values, percentile_min)
    vmax = np.percentile(valid_values, percentile_max)
    if vmax <= vmin:
        normalized = np.zeros(array.shape, dtype=np.uint8)
    else:
        clipped = np.clip(array, vmin, vmax)
        normalized = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    if invert:
        normalized = 255 - normalized
    normalized[~valid_mask] = 0
    return normalized


def resolve_output_dir(input_path: Path, output_dir: Optional[str]) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if input_path.is_file():
        return input_path.parent / "dsm_visual"
    return input_path / "dsm_visual"


def iter_npz_files(input_path: Path):
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.glob("*.npz")):
        yield path


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in iter_npz_files(input_path):
        array = load_array(npz_path, args.key)
        if array.ndim > 2:
            array = np.squeeze(array)
        normalized = normalize_to_uint8(array, args.percentile_min, args.percentile_max, args.invert)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        colored[~np.isfinite(array)] = 0
        output_path = output_dir / f"{npz_path.stem}.png"
        cv2.imwrite(str(output_path), colored)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
