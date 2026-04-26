import argparse
import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.coords import BoundingBox
from pyproj import Transformer, CRS
from pyproj.exceptions import CRSError

# ==============================================================================
# 1. Helper Functions
# ==============================================================================

def get_google_image_bounds_wgs84(image_name, width, height, zoom):
    """
    Estimates the WGS84 bounding box for a Google Static Map API image.
    
    Args:
        image_name (str): The filename, e.g., 'satellite_lat_lon.png'.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        zoom (int): The zoom level used to download the image.

    Returns:
        BoundingBox: The estimated WGS84 bounding box.
    """
    try:
        # Assumes filename format 'prefix_latitude_longitude.png'.
        # We robustly take the last two parts for lat/lon.
        parts = image_name.replace('.png', '').split('_')
        center_lat = float(parts[-2])
        center_lon = float(parts[-1])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not parse latitude/longitude from filename '{image_name}'. Ensure format is 'prefix_lat_lon.png'.") from e

    # Meters per pixel at a given latitude and zoom level for Google Maps
    meters_per_pixel = 156543.03 * np.cos(np.deg2rad(center_lat)) / (2**zoom)

    # Calculate the ground dimensions of the image in meters
    ground_width_m = width * meters_per_pixel
    ground_height_m = height * meters_per_pixel

    # Approximate conversion from meters to degrees of latitude and longitude
    lon_span = (ground_width_m / 1000) / (111.32 * np.cos(np.deg2rad(center_lat)))
    lat_span = (ground_height_m / 1000) / 111.32

    return BoundingBox(
        left=center_lon - lon_span / 2, bottom=center_lat - lat_span / 2,
        right=center_lon + lon_span / 2, top=center_lat + lat_span / 2
    )

def build_dsm_index(dsm_root_dir, tif_names):
    """
    Builds an in-memory index of DSM tile metadata for fast querying.
    This function should be called once at the beginning.
    """
    print("--- Building DSM tile metadata index... ---")
    dsm_index = []
    wgs84_crs = CRS.from_epsg(4326)

    for tif_name in tif_names:
        tif_path = os.path.join(dsm_root_dir, tif_name)
        if not os.path.exists(tif_path):
            print(f"  -> WARNING: File not found, skipping: {tif_path}")
            continue
        try:
            with rasterio.open(tif_path) as src:
                transformer = Transformer.from_crs(src.crs, wgs84_crs, always_xy=True)
                lon_min, lat_min = transformer.transform(src.bounds.left, src.bounds.bottom)
                lon_max, lat_max = transformer.transform(src.bounds.right, src.bounds.top)
                dsm_index.append({
                    "path": tif_path, "bounds_proj": src.bounds,
                    "bounds_wgs84": BoundingBox(left=lon_min, bottom=lat_min, right=lon_max, top=lat_max),
                    "crs": src.crs, "transform": src.transform
                })
                print(f"  -> Indexed: {tif_name}")
        except Exception as e:
            print(f"  -> ERROR indexing file {tif_name}: {e}")
    print(f"--- Index build complete. Indexed {len(dsm_index)} tiles. ---\n")
    return dsm_index

def find_candidates_from_index(dsm_index, target_bounds_wgs84):
    """
    Finds candidate TIFs from the pre-built in-memory index that overlap
    with the target's WGS84 bounding box.
    """
    candidate_tiles = []
    for tile_info in dsm_index:
        dsm_bounds_wgs84 = tile_info["bounds_wgs84"]
        if not (target_bounds_wgs84.right < dsm_bounds_wgs84.left or
                target_bounds_wgs84.left > dsm_bounds_wgs84.right or
                target_bounds_wgs84.top < dsm_bounds_wgs84.bottom or
                target_bounds_wgs84.bottom > dsm_bounds_wgs84.top):
            candidate_tiles.append(tile_info)
    return candidate_tiles

def save_results(dsm_data, base_name, save_dir,save_visualization):
    """
    Saves the final DSM data. The input array can contain NaN values,
    which are handled correctly for both NPZ and PNG outputs.
    """
    # Save the raw elevation data (in meters) to a compressed NPZ file.
    # The NPZ format handles NaN values perfectly.
    output_npz_path = os.path.join(save_dir, base_name + '_dsm.npz')
    np.savez_compressed(output_npz_path, dsm=dsm_data)
    print(f"Saved raw elevation data to: {output_npz_path}")

    # Create a visualization
    valid_mask = ~np.isnan(dsm_data)
    if not np.any(valid_mask):
        print("WARNING: No valid data in the final result. Cannot generate color map image.")
        return

    if save_visualization:
        # Use nan-aware functions to get min/max for normalization
        min_val, max_val = np.nanmin(dsm_data), np.nanmax(dsm_data)
        normalized_dsm = np.zeros_like(dsm_data, dtype=np.uint8)
        
        if max_val > min_val:
            normalized_dsm[valid_mask] = (255 * (dsm_data[valid_mask] - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            normalized_dsm[valid_mask] = 128
        
        colored_dsm = cv2.applyColorMap(normalized_dsm, cv2.COLORMAP_VIRIDIS)
        # Set NoData areas (where mask is False) to black
        colored_dsm[~valid_mask] = [0, 0, 0]

        output_png_path = os.path.join(save_dir, base_name + '_dsm_color.png')
        cv2.imwrite(output_png_path, colored_dsm)
        print(f"Saved visualization to: {output_png_path}")

# ==============================================================================
# 2. Main Execution Logic
# ==============================================================================
def process_single_image(google_image_name, dsm_index, image_width, image_height, zoom_level, save_dir, nan_tolerance,save_visualization):
    try:
        target_bounds_wgs84 = get_google_image_bounds_wgs84(google_image_name, image_width, image_height, zoom_level)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    candidate_tiles = find_candidates_from_index(dsm_index, target_bounds_wgs84)
    if not candidate_tiles:
        print("Processing finished: No overlapping DSM tiles found for: ",google_image_name)
        return

    # print(f"Found {len(candidate_tiles)} candidates: {[os.path.basename(c['path']) for c in candidate_tiles]}")
    
    dsm_crs = candidate_tiles[0]["crs"]
    transformer = Transformer.from_crs(CRS.from_epsg(4326), dsm_crs, always_xy=True)
    google_left_proj, google_bottom_proj = transformer.transform(target_bounds_wgs84.left, target_bounds_wgs84.bottom)
    google_right_proj, google_top_proj = transformer.transform(target_bounds_wgs84.right, target_bounds_wgs84.top)
    dst_transform = from_bounds(google_left_proj, google_bottom_proj, google_right_proj, google_top_proj, width=image_width, height=image_height)
    
    best_dsm_result, max_valid_pixels, best_source_path = None, -1, None

    for tile_info in candidate_tiles:
        tif_path = tile_info["path"]
        temp_dsm = np.full((image_height, image_width), np.nan, dtype=np.float32)
        with rasterio.open(tif_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=temp_dsm, src_transform=src.transform,
                src_crs=src.crs, dst_transform=dst_transform, dst_crs=dsm_crs,
                resampling=Resampling.bilinear, dst_nodata=np.nan
            )
        current_valid_pixels = np.count_nonzero(~np.isnan(temp_dsm))
        if current_valid_pixels > max_valid_pixels:
            max_valid_pixels, best_dsm_result, best_source_path = current_valid_pixels, temp_dsm, tif_path
    

    if max_valid_pixels > 0:
        """
        In metadata, it says:
        "Delivered DEM rasters have pixel size of 1.5 feet, and vertical units of meters. These were converted to feet during intake process and these reprocessed versions are made available through the Washington Lidar Portal."
        """
        # Convert elevation from feet to meters
        FEET_TO_METERS_FACTOR = 0.304800609601219
        dsm_in_meters = best_dsm_result.copy()
        valid_mask = ~np.isnan(dsm_in_meters)
        dsm_in_meters[valid_mask] *= FEET_TO_METERS_FACTOR
        
        # --- NEW: Apply NaN percentage policy ---
        nan_count = np.count_nonzero(np.isnan(dsm_in_meters))
        total_pixels = dsm_in_meters.size
        nan_percentage = (nan_count / total_pixels) * 100
        
        
        if nan_percentage > nan_tolerance:
            print(f"POLICY: Discarding image because NaN percentage ({nan_percentage:.2f}%) exceeds tolerance ({nan_tolerance:.2f}%).")
            return
            
        final_dsm = dsm_in_meters
        output_base_name = google_image_name.replace('.png', '')
        save_results(final_dsm, output_base_name, save_dir,save_visualization)
        
    else:
        print("WARNING: The final cropped area contains no valid DSM data.")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Seattle DSM tiles for a VIGOR split.")
    parser.add_argument("--dsm_root_dir", type=str, required=True, help="Directory containing the raw DSM GeoTIFF tiles.")
    parser.add_argument("--split_txt", type=str, required=True, help="VIGOR split file used to select satellite images.")
    parser.add_argument("--save_dir", type=str, default="./Seattle_DSM", help="Directory used to save the extracted DSM results.")
    parser.add_argument(
        "--dsm_tif_names",
        nargs="+",
        default=["king_county_west_2021_dsm_87.tif", "king_county_west_2021_dsm_88.tif", "king_county_west_2021_dsm_89.tif", "king_county_west_2021_dsm_90.tif", "king_county_west_2021_dsm_111.tif", "king_county_west_2021_dsm_112.tif"],
        help="GeoTIFF tile names under `dsm_root_dir`.",
    )
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=640)
    parser.add_argument("--zoom_level", type=int, default=20)
    parser.add_argument("--nan_tolerance_percentage", type=float, default=5.0)
    parser.add_argument("--save_visualization", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    SATELLITE_IMAGES_TO_PROCESS = []
    with open(args.split_txt) as f:
        for line in f:
            SATELLITE_IMAGES_TO_PROCESS.append(os.path.basename(line.strip().split(' ')[0]))
    print(len(SATELLITE_IMAGES_TO_PROCESS))
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"All outputs will be saved to: {os.path.abspath(args.save_dir)}")

    dsm_index = build_dsm_index(args.dsm_root_dir, args.dsm_tif_names)
    if not dsm_index:
        print("ERROR: Failed to build DSM index. Exiting.")
        return

    for image_name in SATELLITE_IMAGES_TO_PROCESS:
        process_single_image(
            image_name, dsm_index, 
            args.image_width, args.image_height, args.zoom_level,
            args.save_dir, args.nan_tolerance_percentage, args.save_visualization
        )
        
if __name__ == '__main__':
    main()
