import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, shift, rotate
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform, estimate_transform, warp

# ==============================================================================
# 1. UPDATED Spatial Alignment and Core Functions
# ==============================================================================

def normalize_to_uint8(arr, vmin=None, vmax=None):
    """
    Normalizes a floating-point array to a uint8 grayscale image.
    If vmin and vmax are provided, they are used for normalization.
    Otherwise, range is calculated from the array's 2nd/98th percentiles.
    """
    arr_valid = arr[~np.isnan(arr)]
    if arr_valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    if vmin is None or vmax is None:
        min_val, max_val = np.percentile(arr_valid, 2), np.percentile(arr_valid, 98)
    else:
        min_val, max_val = vmin, vmax
    
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.uint8)

    arr_clipped = np.clip(arr, min_val, max_val)
    arr_normalized = 255 * (arr_clipped - min_val) / (max_val - min_val)
    arr_normalized[np.isnan(arr_normalized)] = 0
    return arr_normalized.astype(np.uint8)

def align_spatially_sift_rigid(ref_map, moving_map, max_shift, max_rotation):
    """
    A SIFT-based fallback that estimates a RIGID transform and then VALIDATES it against constraints.
    Returns the aligned map if valid, otherwise returns None.
    """
    try:
        ref_gray, moving_gray = normalize_to_uint8(ref_map), normalize_to_uint8(moving_map)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ref_gray, None)
        kp2, des2 = sift.detectAndCompute(moving_gray, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            raise ValueError("Not enough descriptors found.")
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) < 10:
            raise ValueError("Not enough good matches after ratio test.")
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Use scikit-image to estimate a rigid (Euclidean) transform
        tform = estimate_transform('euclidean', src_pts[:, ::-1], dst_pts[:, ::-1])
        if tform is None:
            raise ValueError("scikit-image could not estimate a valid transform.")

        # --- NEW: VALIDATION STEP ---
        # Extract translation and rotation from the estimated transform
        translation_y, translation_x = tform.translation
        rotation_deg = np.rad2deg(tform.rotation)
        
        print(f"SIFT Fallback Estimated: Shift (y,x)=({translation_y:.2f}, {translation_x:.2f}), Rotation={rotation_deg:.2f} deg")
        
        # Check if the estimated transform exceeds the allowed constraints
        if abs(translation_y) > max_shift or abs(translation_x) > max_shift or abs(rotation_deg) > max_rotation:
            print(f"VALIDATION FAILED: Estimated transform exceeds constraints (shift > {max_shift} or rotation > {max_rotation}).")
            return moving_map # Return None to signal failure

        print("VALIDATION PASSED: Estimated transform is within constraints.")
        
        # Apply the validated transform
        aligned_map = warp(moving_map, tform.inverse, output_shape=ref_map.shape,
                           mode='constant', cval=np.nan, preserve_range=True, order=1)
        
        return aligned_map.astype(moving_map.dtype)

    except Exception as e:
        print(f"CRITICAL: Rigid SIFT fallback process failed: {e}.",'use the raw img')
        return moving_map

def align_spatially_hybrid(ref_map, moving_map, max_shift=20, max_rotation=10):
    """A hybrid spatial alignment strategy that strictly adheres to constraints."""
    try:
        ref_norm, moving_norm = normalize_to_uint8(ref_map), normalize_to_uint8(moving_map)
        shifts, _, _ = phase_cross_correlation(ref_norm, moving_norm, upsample_factor=10)
        y_shift, x_shift = shifts
        
        if abs(y_shift) > max_shift or abs(x_shift) > max_shift:
            raise ValueError(f"Phase correlation shift ({y_shift:.1f}, {x_shift:.1f}) exceeds max_shift={max_shift}px.")
        print(f"Found translation shift (y, x): ({y_shift:.2f}, {x_shift:.2f})")
        
        best_angle, max_corr = 0.0, -1.0
        map_shifted_for_search = shift(moving_norm, shift=shifts, mode='constant', cval=0, order=1)
        for angle in np.arange(-max_rotation, max_rotation + 0.5, 0.5):
            map_rotated = rotate(map_shifted_for_search, angle, reshape=False, mode='constant', cval=0, order=1)
            valid_overlap = (ref_norm > 0) & (map_rotated > 0)
            if np.count_nonzero(valid_overlap) < 100: continue
            corr = np.corrcoef(ref_norm[valid_overlap], map_rotated[valid_overlap])[0, 1]
            if corr > max_corr: max_corr, best_angle = corr, angle
        print(f"Found best rotation angle: {best_angle:.2f} degrees with correlation {max_corr:.4f}")
        
        map_shifted_hires = shift(moving_map, shift=shifts, mode='constant', cval=np.nan, order=1)
        return rotate(map_shifted_hires, best_angle, reshape=False, mode='constant', cval=np.nan, order=1)
        
    except Exception as e:
        print(f"Hybrid alignment failed: {e}. Falling back to constrained RIGID SIFT.")
        return align_spatially_sift_rigid(ref_map, moving_map, max_shift, max_rotation)

def align_vertically_with_ransac(pred_map, dsm_aligned, n_iterations=150, sample_size=50, inlier_threshold=1.5):
    """Finds the global vertical offset (h_offset) between two maps using RANSAC."""
    best_inlier_count, best_h_offset = -1, 0
    valid_mask = ~np.isnan(pred_map) & ~np.isnan(dsm_aligned)
    valid_indices = np.argwhere(valid_mask)
    if valid_indices.shape[0] < sample_size: raise RuntimeError("Not enough overlapping data for RANSAC.")
    for i in range(n_iterations):
        random_indices = valid_indices[np.random.choice(len(valid_indices), sample_size, replace=False)]
        sample_coords = tuple(random_indices.T)
        h_candidate = np.median(dsm_aligned[sample_coords] - pred_map[sample_coords])
        error = np.abs((dsm_aligned[valid_mask] - pred_map[valid_mask]) - h_candidate)
        inlier_count = np.count_nonzero(error < inlier_threshold)
        if inlier_count > best_inlier_count: best_inlier_count, best_h_offset = inlier_count, h_candidate
    print(f"RANSAC found a model with {best_inlier_count} inliers.")
    final_error = np.abs((dsm_aligned[valid_mask] - pred_map[valid_mask]) - best_h_offset)
    final_inlier_mask_flat = final_error < inlier_threshold
    final_inlier_indices = valid_indices[final_inlier_mask_flat]
    if len(final_inlier_indices) < 20:
        print("WARNING: Final inlier count is low.")
        return best_h_offset
    final_inlier_coords = tuple(final_inlier_indices.T)
    h_offset_final = np.mean(dsm_aligned[final_inlier_coords] - pred_map[final_inlier_coords])
    print(f"Final refined vertical offset (h_offset): {h_offset_final:.3f} meters")
    return h_offset_final

def visualize_single_result(pred_map, dsm_fully_aligned, output_dir, base_name):
    """Saves a comprehensive set of comparison images, with an absolute error map."""
    print("\n--- Generating Visualization Images for Single Result ---")
    
    # --- Prediction and Aligned GT Visualization (remains the same) ---
    valid_mask = ~np.isnan(pred_map) & ~np.isnan(dsm_fully_aligned)
    shared_vmin, shared_vmax = None, None
    if np.count_nonzero(valid_mask) > 0:
        combined_data = np.concatenate([pred_map[valid_mask], dsm_fully_aligned[valid_mask]])
        shared_vmin, shared_vmax = np.percentile(combined_data, 2), np.percentile(combined_data, 98)
        print(f"Using shared visualization range (vmin, vmax): ({shared_vmin:.2f}, {shared_vmax:.2f})")
    
    pred_viz = cv2.applyColorMap(normalize_to_uint8(pred_map, vmin=shared_vmin, vmax=shared_vmax), cv2.COLORMAP_VIRIDIS); pred_viz[np.isnan(pred_map)] = 0
    cv2.putText(pred_viz, "Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    gt_viz = cv2.applyColorMap(normalize_to_uint8(dsm_fully_aligned, vmin=shared_vmin, vmax=shared_vmax), cv2.COLORMAP_VIRIDIS); gt_viz[np.isnan(dsm_fully_aligned)] = 0
    cv2.putText(gt_viz, "Aligned GT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # ==================== ERROR MAP LOGIC CHANGE ====================
    # 1. Calculate the ABSOLUTE error
    abs_error_map = np.abs(pred_map - dsm_fully_aligned)
    
    # 2. Clip the absolute error from 0 to 5 meters
    error_viz_clipped = np.clip(abs_error_map, 0, 5)
    
    # 3. Normalize and apply the HOT colormap (Black -> Red -> Yellow -> White)
    error_viz_color = cv2.applyColorMap(normalize_to_uint8(error_viz_clipped), cv2.COLORMAP_HOT)
    error_viz_color[np.isnan(abs_error_map)] = 0 # Black out NaN areas
    
    # 4. Update the text label
    cv2.putText(error_viz_color, "Abs Error (0-5m+)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    # ================================================================
    
    summary_image = cv2.hconcat([pred_viz, gt_viz, error_viz_color])
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_00_comparison_summary.png"), summary_image)
    print(f"Saved visualization images to '{output_dir}'.")


def plot_pixel_error_histogram(all_pixel_errors, output_dir, max_error_val=50.0, bin_width=1.0):
    """Plots a histogram of pixel-wise errors, with a special bin for errors > max_error_val."""
    if all_pixel_errors.size == 0: print("\nNo pixel errors collected."); return
    print("\n--- Generating Overall Pixel-Level Error Distribution Plot ---"); total_pixels = len(all_pixel_errors); bins = np.append(np.arange(0, max_error_val + bin_width, bin_width), np.inf)
    counts, _ = np.histogram(all_pixel_errors, bins=bins); percentages = counts / total_pixels * 100
    plt.figure(figsize=(15, 8)); bar_positions = np.arange(len(percentages)); plt.bar(bar_positions, percentages, width=0.9, edgecolor='black', alpha=0.75)
    tick_labels = [f"{int(b)}" for b in bins[:-2]] + [f">{int(max_error_val)}"]; display_interval = 5; display_positions = list(range(0, len(tick_labels)-1, display_interval)) + [len(tick_labels)-1]
    plt.xticks(np.array(bar_positions)[display_positions], [tick_labels[i] for i in display_positions], rotation=45)
    plt.title('Overall Pixel-Level Error Distribution', fontsize=18, fontweight='bold'); plt.xlabel('Absolute Error (meters)', fontsize=14); plt.ylabel('Percentage of Total Pixels (%)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6); plt.xlim(-0.5, len(bar_positions) - 0.5)
    mean_abs_error = np.mean(all_pixel_errors); median_abs_error = np.median(all_pixel_errors); p_lt_1m = np.count_nonzero(all_pixel_errors < 1.0) / total_pixels * 100; p_lt_3m = np.count_nonzero(all_pixel_errors < 3.0) / total_pixels * 100
    stats_text = (f"Total Pixels: {total_pixels:,}\n\n"f"Overall MAE: {mean_abs_error:.3f} m\n"f"Median AE: {median_abs_error:.3f} m\n\n"f"Pixels with error < 1m: {p_lt_1m:.2f}%\n"f"Pixels with error < 3m: {p_lt_3m:.2f}%")
    plt.text(0.97, 0.97, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))
    plot_path = os.path.join(output_dir, "overall_pixel_error_distribution.png"); plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close(); print(f"Saved pixel-level error distribution plot to: {plot_path}")


def _plot_metric_histogram(metric_list, metric_name, output_dir, color, max_val=40.0, bin_width=2.5):
    """
    Generic function to plot a histogram for a given metric (RMSE or MAE).
    This version plots on a real-valued axis for correct alignment.
    """
    if not metric_list:
        print(f"\nNo {metric_name} values collected to plot histogram.")
        return

    # 1. Define the histogram bins.
    bins = np.append(np.arange(0, max_val + bin_width, bin_width), np.inf)

    # 2. Count samples in each bin and convert them to percentages.
    counts, _ = np.histogram(metric_list, bins=bins)
    total_items = len(metric_list)
    percentages = (counts / total_items) * 100 if total_items > 0 else np.zeros_like(counts)

    # --- Plotting logic ---
    plt.figure(figsize=(15, 8))
    
    # 3. Use the real bin edges on the x-axis and draw bars from the left edge.
    # Keep a small gap between bars for readability.
    bar_width = bin_width * 0.9
    bar_positions = bins[:-1] + (bin_width * 0.05) # Slightly center each bar inside its interval.
    plt.bar(bar_positions, percentages, width=bar_width, align='edge', edgecolor='black', alpha=0.75, color=color)

    # 4. Create x-axis ticks that match the real-valued axis.
    display_interval = 2  # Show one tick every few bins.
    tick_positions = np.arange(0, max_val + bin_width, display_interval * bin_width)
    tick_labels = [f"{int(pos)}" for pos in tick_positions]
    
    # Rename the last tick to highlight overflow values.
    if tick_positions[-1] == max_val:
        tick_labels[-1] = f">{int(max_val)}"
    
    plt.xticks(tick_positions, tick_labels, fontsize=12)

    # 5. Draw the mean line at the true metric value.
    mean_metric = np.mean(metric_list)
    plt.axvline(mean_metric, color='r', linestyle='--', linewidth=2, label=f'Mean {metric_name}: {mean_metric:.3f}')

    # --- Final formatting and saving ---
    plt.title(f'Distribution of Per-Image {metric_name}', fontsize=18, fontweight='bold')
    plt.xlabel(f'{metric_name} (meters)', fontsize=14)
    plt.ylabel('Percentage of Images (%)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Keep a small margin on the left side of the x-axis.
    plt.xlim(left=-1, right=max_val + bin_width)
    plt.ylim(bottom=0)

    plot_path = os.path.join(output_dir, f"per_image_{metric_name.lower()}_distribution_percent.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-image {metric_name} percentage distribution plot to: {plot_path}")

def plot_rmse_histogram(rmse_list, output_dir, max_val=40.0, bin_width=2.5):
    """Plots a histogram of the per-image RMSE values."""
    _plot_metric_histogram(
        metric_list=rmse_list,
        metric_name='RMSE',
        output_dir=output_dir,
        color='steelblue', # Default color for RMSE.
        max_val=max_val,
        bin_width=bin_width
    )

def plot_mae_histogram(mae_list, output_dir, max_val=40.0, bin_width=2.5):
    """Plots a histogram of the per-image MAE values."""
    _plot_metric_histogram(
        metric_list=mae_list,
        metric_name='MAE',
        output_dir=output_dir,
        color='mediumseagreen', # Default color for MAE.
        max_val=max_val,
        bin_width=bin_width
    )



# ==============================================================================
# 3. Main Workflow with Caching and Strict Alignment
# ==============================================================================

def single_evaluation_run(pred_npz_path, dsm_npz_path, output_dir, spatial_cache_dir, visualize=False,spatial_align=True):
    """
    Performs a full evaluation run for a single pair, now with strict, constrained alignment.
    """
    base_name = os.path.splitext(os.path.basename(pred_npz_path))[0]
    gt_base_name = os.path.splitext(os.path.basename(dsm_npz_path))[0].replace('_dsm', '')
    
    try:
        pred_map = np.load(pred_npz_path)['arr_0']; dsm_map_original = np.load(dsm_npz_path)['dsm']
        # pred_map = pred_map*74/64.48 # for checking spatial resolution
        pred_map  = pred_map * 64.48
        pred_h, pred_w = pred_map.shape; dsm_for_alignment = dsm_map_original
        if dsm_map_original.shape != (pred_h, pred_w):
            dsm_for_alignment = cv2.resize(dsm_map_original, dsize=(pred_w, pred_h), interpolation=cv2.INTER_LINEAR)
        
        if spatial_align:
            cache_filename = f"{gt_base_name}.npz"; cached_dsm_path = os.path.join(spatial_cache_dir, cache_filename)
            
            if os.path.exists(cached_dsm_path):
                print(f"Found cached spatially aligned DSM. Loading from: {cached_dsm_path}"); dsm_spatially_aligned = np.load(cached_dsm_path)['dsm']
            else:
                dsm_spatially_aligned = align_spatially_hybrid(ref_map=pred_map, moving_map=dsm_for_alignment)
                
                # --- NEW: Handle alignment failure ---
                if dsm_spatially_aligned is None:
                    raise RuntimeError("Spatial alignment failed because no valid transform within constraints was found.")
                
                print(f"Saving spatially aligned DSM to cache: {cached_dsm_path}"); np.savez_compressed(cached_dsm_path, dsm=dsm_spatially_aligned)
        else: 
            dsm_spatially_aligned = dsm_for_alignment
        h_offset = align_vertically_with_ransac(pred_map, dsm_spatially_aligned)
        if np.isnan(h_offset): raise RuntimeError("Vertical alignment (RANSAC) failed.")
        
        dsm_fully_aligned = dsm_spatially_aligned - h_offset
        final_error_map = np.abs(pred_map - dsm_fully_aligned)
        valid_errors = final_error_map[~np.isnan(final_error_map)]
        if valid_errors.size == 0: 
            raise RuntimeError("No valid overlapping data after alignment.")
        
        mae_single_image = np.mean(valid_errors)
        rmse_single_image = np.sqrt(np.mean(valid_errors**2)); print(f"\nMetrics for this image: MAE = {mae_single_image:.4f} m, RMSE = {rmse_single_image:.4f} m")

        if visualize:
            visualize_single_result(pred_map, dsm_fully_aligned, output_dir, base_name)
            
        return valid_errors,rmse_single_image,mae_single_image
        
    except Exception as e:
        print(f"CRITICAL ERROR during processing of {base_name}: {e}"); return None

def main(pred_path, gt_dsm_path, output_directory, spatial_align_cache_dir):
    DEBUG = False
    SPATIAL_ALIGN = True
    print('spatial_align: ', SPATIAL_ALIGN)
    assert SPATIAL_ALIGN
    os.makedirs(output_directory, exist_ok=True)
    pred_path = str(Path(pred_path))
    base_checkpoint_name = Path(pred_path).parent.name
    OUTPUT_DIRECTORY = os.path.join(output_directory, base_checkpoint_name)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(spatial_align_cache_dir, exist_ok=True)

    file_pairs = []
    pred_DSM_list = os.listdir(pred_path)
    gt_DSM_list = os.listdir(gt_dsm_path)
    for pred_DSM in pred_DSM_list:
        if pred_DSM.replace('.npz', '_dsm.npz') in gt_DSM_list:
            file_pairs.append((os.path.join(pred_path, pred_DSM), os.path.join(gt_dsm_path, pred_DSM.replace('.npz', '_dsm.npz'))))
    if DEBUG:
        file_pairs = file_pairs[:1]

    # --- INITIALIZE LISTS ---
    all_pixel_errors_list = []
    rmse_list = []
    mae_list = []
    
    should_visualize_single_runs = (len(file_pairs) == 1)
    
    for i, (pred_path, dsm_path) in enumerate(file_pairs):
        print(f"\n{'='*25} Processing: {i+1}/{len(file_pairs)} {os.path.basename(pred_path)} {'='*25}")
        if not (os.path.exists(pred_path) and os.path.exists(dsm_path)):
            print(f"WARNING: Skipping pair, file not found: {pred_path} or {dsm_path}"); continue
            
        result = single_evaluation_run(pred_path, dsm_path, OUTPUT_DIRECTORY, spatial_align_cache_dir, visualize=should_visualize_single_runs, spatial_align=SPATIAL_ALIGN)
        
        if result is not None:
            pixel_errors, rmse_single_image, mae_single_image = result
            # Append all collected metrics
            all_pixel_errors_list.append(pixel_errors)
            rmse_list.append(rmse_single_image)
            mae_list.append(mae_single_image)

    # --- FINAL AGGREGATE METRICS CALCULATION AND PLOTTING ---
    if not all_pixel_errors_list:
        print("\nNo successful evaluations were completed. No metrics to report.")
        return

    # --- 1. Calculate overall per-pixel metrics (<2.5m, <7.5m, etc.) ---
    print("\n" + "="*30 + " OVERALL DATASET METRICS " + "="*30)
    all_pixel_errors = np.concatenate(all_pixel_errors_list)
    total_pixels = len(all_pixel_errors)
    
    # Calculate percentages for thresholds
    p_lt_2_5 = (np.count_nonzero(all_pixel_errors < 2.5) / total_pixels) * 100
    p_lt_7_5 = (np.count_nonzero(all_pixel_errors < 7.5) / total_pixels) * 100

    # --- 2. Calculate average per-image metrics ---
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    
    # --- 3. Print all summary metrics together ---
    print(f"Average Per-Image MAE:  {avg_mae:.4f}")
    print(f"Average Per-Image RMSE: {avg_rmse:.4f}")
    print(f"Overall <2.5m (%):      {p_lt_2_5:.2f}")
    print(f"Overall <7.5m (%):      {p_lt_7_5:.2f}")
    print("="*82)

    # --- 4. Save aggregated data and generate plots ---
    
    # Save all pixel errors for potential later analysis
    errors_save_path = os.path.join(OUTPUT_DIRECTORY, "overall_pixel_errors.npz")
    np.savez_compressed(errors_save_path, errors=all_pixel_errors)
    print(f"\nSaved all collected pixel errors to: {errors_save_path}")
    
    # Generate overall pixel error histogram (this function already calculates and shows MAE/Median etc.)
    plot_pixel_error_histogram(all_pixel_errors, OUTPUT_DIRECTORY)

    # Save per-image lists
    np.savez_compressed(os.path.join(OUTPUT_DIRECTORY, "per_image_rmse.npz"), rmse=rmse_list)
    np.savez_compressed(os.path.join(OUTPUT_DIRECTORY, "per_image_mae.npz"), mae=mae_list)
    
    # Generate per-image distribution plots
    plot_rmse_histogram(rmse_list, OUTPUT_DIRECTORY)
    # ** BUG FIX: Pass mae_list to the MAE plotting function, not rmse_list **
    plot_mae_histogram(mae_list, OUTPUT_DIRECTORY)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate predicted DSM `.npz` files against Seattle DSM ground truth.")
    parser.add_argument('--pred_path', type=str, required=True, help="Directory containing predicted DSM `.npz` files.")
    parser.add_argument('--gt_dsm_path', type=str, required=True, help="Directory containing ground-truth Seattle DSM `.npz` files.")
    parser.add_argument('--output_dir', type=str, default='output_remote/DSM_metric_results')
    parser.add_argument('--spatial_align_cache_dir', type=str, default='./cached_spatial_aligned_DSM')
    args = parser.parse_args()
    main(args.pred_path, args.gt_dsm_path, args.output_dir, args.spatial_align_cache_dir)
