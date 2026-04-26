# Evaluation

## Evaluation vs Inference

`inference/` generates outputs from a checkpoint.

`evaluation` in this release means **dataset-level quantitative scoring**. That includes:

- image metrics
- exporting predicted DSM files for a split
- preparing Seattle DSM ground truth
- absolute DSM evaluation

The checkpoint-based evaluation path assumes a CUDA-enabled environment.

## Released Checkpoint Metrics

The following metrics were obtained using the released checkpoint on the VIGOR test split:

| Metric | Value |
|--------|-------|
| **PSNR** | 13.846 |
| **SSIM** | 0.372 |
| **LPIPS (AlexNet)** | 0.381 |
| **LPIPS (SqueezeNet)** | 0.287 |
| **FID** | 19.029 |
| **KID** | 0.014 |

## 1. Image Metrics

```bash
accelerate launch --num_processes 1 inference/evaluate_img_metrics.py \
  --data_root data/vigor \
  --test_split test \
  --sky_from_training
```

For quick testing with a limited number of samples:

```bash
accelerate launch --num_processes 1 inference/evaluate_img_metrics.py \
  --data_root data/vigor \
  --test_split test \
  --sky_from_training \
  --max_samples 30
```

This is the released dataset-level image evaluation entry.

It evaluates generated street-view outputs over a split. The released pipeline keeps the standard image metrics (PSNR, SSIM, LPIPS, FID, KID) and does **not** include the DINO semantic metric.

## 2. Export Predicted DSM Files

```bash
accelerate launch --num_processes 1 inference/evaluate_img_metrics.py \
  --data_root YOUR_VIGOR_ROOT \
  --test_split test \
  --save_DSM \
  --sky_from_training \
  --if_save_image False
```

This command exports model-predicted DSM files to:

- `.../DSM_result/*.npz`

This step is **not** the final DSM metric by itself. It prepares the prediction files used by the DSM evaluation script below.

## 3. Prepare Seattle DSM Ground Truth

```bash
python DSM_processing/processing_DSM_pair_from_txt.py \
  --dsm_root_dir path/to/raw_seattle_dsm_tiles \
  --split_txt YOUR_VIGOR_ROOT/test_remove_building.txt \
  --save_dir YOUR_VIGOR_ROOT/Seattle_DSM
```

This script prepares the ground-truth Seattle DSM files for the released split.

## 4. Absolute DSM Evaluation

```bash
python DSM_processing/calculate_DSM_metric2.py \
  --pred_path output_remote/save_result/vigor/test/your_model/checkpoint-xxxxxx/DSM_result \
  --gt_dsm_path YOUR_VIGOR_ROOT/Seattle_DSM
```

This is the **main DSM metric** for the released model predictions.

Input:

- predicted DSM `.npz` files exported from `inference/evaluate_img_metrics.py --save_DSM`
- ground-truth Seattle DSM `.npz` files under `Seattle_DSM/`

What it does:

- spatially aligns the ground-truth DSM to the prediction
- caches the aligned DSM in `--spatial_align_cache_dir`
- estimates a vertical offset
- reports MAE, RMSE, and threshold-based statistics

If you want the released DSM benchmark, use **Absolute DSM Evaluation** above.
