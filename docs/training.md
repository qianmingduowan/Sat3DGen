# Training

## Scope

The public release keeps the cleaned VIGOR training path used in this repository.

The released training path assumes a CUDA-enabled environment.

Main entry points:

- `train.py`
- `demo_train.sh`
- `configs/Sat3DGen_dino_v3.json`

## Quick Start

```bash
bash demo_train.sh
```

Run the launcher from the repository root so that `train.py` and `configs/` resolve correctly.

Before launching training, set:

- `DATA_DIR`: prepared VIGOR root
- `JOB_NAME`: experiment name
- `OUT_BASE_DIR`: optional output root override
- `DATA_TXT`: optional split-file override

## Main Launcher Variables

The released launcher reads:

- `DATA_DIR`
- `JOB_NAME`
- `OUT_BASE_DIR`
- `TRAIN_BATCH_SIZE`
- `NUM_PROCESSES`
- `DATA_TXT`
- `CONF_DIR`

Example:

```bash
DATA_DIR=/path/to/VIGOR \
JOB_NAME=sat3dgen_release \
OUT_BASE_DIR=./output \
bash demo_train.sh
```

## Reference Training Setup

The released checkpoint was trained with the following empirical setup (not necessarily optimal, provided for reproducibility):

- **GPUs**: 24 × A100 (or equivalent)
- **Per-GPU batch size**: 2 (`--train_batch_size 2`)
- **Total iterations**: ~600,000
- **Epoch schedule**: `--num_train_epochs 32` × internal multiplier 16 = 512 logical epochs
- **Optimizer**: Adam with `--adam_beta1 0.0 --adam_beta2 0.9`
- **EMA**: enabled (`--use_ema`)
- **LPIPS**: panorama-only LPIPS reconstruction loss (default behaviour)

These hyperparameters are reflected in the default `demo_train.sh`. You may need to adjust `NUM_PROCESSES` and `TRAIN_BATCH_SIZE` to fit your hardware. The total effective batch size is `NUM_PROCESSES × TRAIN_BATCH_SIZE`.

## Notes

- Training-time validation has been removed from the public release.
- The released training path keeps the raw-view auxiliary loss enabled internally.
- The default released training split is `train__corrected_all_3city_remove_building.txt`.
- `train.py` applies an empirical epoch multiplier of `16` to `--num_train_epochs`. Treat the launcher value as the base schedule before this preserved multiplier.

## Config Notes

For important released config fields, see [config_notes.md](config_notes.md).
