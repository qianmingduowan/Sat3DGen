import argparse
import os
from pathlib import Path

from accelerate import Accelerator
from torch.utils.data import DataLoader

from metrics.pair_metric_s2d import pair_metric_sat2density
from my_datasets.vigor import vigor_dataset
from source.generator import Sat3DGen


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Sat3DGen checkpoint on VIGOR.")
    parser.add_argument("--checkpoint_path", type=str, default="qian43/Sat3DGen", help="Model path: HuggingFace repo id or local checkpoint directory.")
    parser.add_argument("--data_root", type=str, required=True, help="Prepared VIGOR dataset root.")
    parser.add_argument("--output_dir", type=str, default="./results/evaluate")
    parser.add_argument("--if_save_image", type=str_to_bool, default=True)
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        choices=["train", "test", "val", "val_origin", "val_noremove"],
    )
    parser.add_argument("--split_txt", type=str, default=None, help="Optional explicit split file path.")
    parser.add_argument("--score_test", action="store_true", default=True, help="Compute and print evaluation metrics (PSNR, SSIM, LPIPS).")
    parser.add_argument("--cat_img2save", action="store_true")
    parser.add_argument("--sky_from_training", action="store_true")
    parser.add_argument("--save_DSM", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit the number of samples for quick testing.")
    return parser.parse_args()


HUGGINGFACE_REPO = "qian43/Sat3DGen"

def resolve_checkpoint_path(checkpoint_root):
    """Locate the model weights directory or fall back to HuggingFace.

    Accepts three layouts:
    1. ``checkpoint_root`` itself contains ``config.json`` (released weights).
    2. ``checkpoint_root/vqmodel_ema`` exists (training checkpoint with EMA).
    3. ``checkpoint_root/vqmodel`` exists (training checkpoint without EMA).
    4. None of the above → return HuggingFace repo id for auto-download.
    """
    checkpoint_root = Path(checkpoint_root)
    if checkpoint_root.name in {"vqmodel", "vqmodel_ema"}:
        raise ValueError("Please pass the checkpoint directory, not `vqmodel` or `vqmodel_ema`.")

    if (checkpoint_root / "config.json").exists():
        return str(checkpoint_root)

    ema_path = checkpoint_root / "vqmodel_ema"
    if ema_path.exists():
        return str(ema_path)

    model_path = checkpoint_root / "vqmodel"
    if model_path.exists():
        return str(model_path)

    print(f"[model] Local checkpoint not found at '{checkpoint_root}', will load from HuggingFace: {HUGGINGFACE_REPO}")
    return HUGGINGFACE_REPO


def get_split_txt(data_root, split_name):
    split_to_file = {
        "train": "train__corrected_all_3city_remove_building.txt",
        "test": "test_remove_building.txt",
        "val": "val__corrected_remove_building.txt",
        "val_origin": "val.txt",
        "val_noremove": "val__corrected.txt",
    }
    return os.path.join(data_root, split_to_file[split_name])


def build_model(args, checkpoint_path):
    Sat3DGen._skip_backbone_weights = True
    model = Sat3DGen.from_pretrained(checkpoint_path)
    Sat3DGen._skip_backbone_weights = False
    patch_size = model.unet_model.patch_size if hasattr(model.unet_model, "patch_size") else 16
    return model, patch_size


def build_dataset(args, model, patch_size, data_txt):
    return vigor_dataset(
        render_size=256,
        sr_factor=model.sr_factor,
        root=args.data_root,
        is_train=False,
        data_txt=data_txt,
        input_resize=patch_size * 16,
    )


def get_batch_size(args):
    if args.save_DSM:
        return 10
    if args.cat_img2save:
        return 6
    if args.score_test or args.if_save_image:
        return 15
    return 4


def get_output_dir(args):
    checkpoint_root = Path(args.checkpoint_path.rstrip("/"))
    # For HuggingFace repo ids like "qian43/Sat3DGen", use "Sat3DGen" as the
    # model name instead of the user/org prefix.
    if "/" in args.checkpoint_path and not checkpoint_root.exists():
        model_name = "Sat3DGen"
    else:
        model_name = checkpoint_root.parent.name
    checkpoint_name = checkpoint_root.name
    return Path(args.output_dir) / "vigor" / args.test_split / model_name / checkpoint_name


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    print(f"Loading model from {checkpoint_path}")

    output_dir = get_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, patch_size = build_model(args, checkpoint_path)
    data_txt = args.split_txt if args.split_txt is not None else get_split_txt(args.data_root, args.test_split)
    dataset = build_dataset(args, model, patch_size, data_txt)

    if args.max_samples is not None and args.max_samples < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(args.max_samples))
        print(f"Limiting dataset to {args.max_samples} samples for quick testing.")

    dataloader = DataLoader(dataset, batch_size=get_batch_size(args), shuffle=False, num_workers=8)

    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    metric_function = pair_metric_sat2density(
        work_dir=str(output_dir),
        save_img=args.if_save_image,
        data_path=args.data_root,
        save_sat=False,
        save_per=False,
        two_view_input_mode=False,
        cat_img2save=args.cat_img2save,
        score_test=args.score_test,
        sky_from_training=args.sky_from_training,
        save_DSM=args.save_DSM,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Dataloader batches: {len(dataloader)}")
    metric_function.evaluate(dataloader, model, accelerator)
