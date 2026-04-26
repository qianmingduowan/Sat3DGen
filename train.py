# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ.setdefault('NCCL_DEBUG', 'ERROR')

import argparse
import math
import shutil
import time
from pathlib import Path
from easydict import EasyDict as edict
import datetime

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from source.discriminators import Discriminator
from source.generator import Sat3DGen
from source.losses.depth_loss import MidasLoss
from source.losses.density_tv_loss import compute_density_tv_loss
from source.losses.sat_depth_loss import compute_depth_loss
from source.training_utils import save_checkpoint
from my_datasets.vigor import vigor_dataset
from huggingface_hub import create_repo
from packaging import version
from tqdm import tqdm

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
import lpips

import random

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

def find_last_checkpoint(output_dir):
    dirs = os.listdir(output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None
    path = os.path.join(output_dir, path) if path is not None else None
    return path


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def gradient_penalty(images, output, weight=10):
    gradients = torch.autograd.grad(
        outputs=output.sum(),
        inputs=images,
        # grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    bsz = gradients.shape[0]
    gradients = torch.reshape(gradients, (bsz, -1))
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def mask_l1_loss(pred, gt, mask):
    mask_sum = mask.sum() * 3
    loss = (torch.abs(pred - gt) * mask).sum() / (mask_sum + 1e-8)
    return loss


def Custom_non_sky_MeanLoss(pred, gt, mask):
    _, _, h, w = pred.shape
    assert gt.min() >= 0, "gt min value should be >= 0"
    non_zero_and_non_sky_pred = ((pred.sum(dim=1) >= 0.03) * (mask.squeeze(1) == 0)).unsqueeze(1)

    valid_mask = ((mask.sum(dim=(2, 3)) / (1 * h * w)) >= 0.1)
    mask_gt = ((gt * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)).unsqueeze(-1).unsqueeze(-1)
    mask_gt = mask_gt.expand_as(pred)
    mask_gt = mask_gt * non_zero_and_non_sky_pred
    masked_pred = pred * non_zero_and_non_sky_pred
    diff = torch.abs(masked_pred - mask_gt).sum(dim=(2, 3)) / (non_zero_and_non_sky_pred.sum(dim=(2, 3)) + 1e-8)
    final_loss = (valid_mask * diff).sum() / (valid_mask.expand_as(diff).sum() + 1e-8)
    return final_loss



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--log_grad_norm_steps",
        type=int,
        default=500,
        help=("Print logs of gradient norms every X steps."),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=51,
        help=("Print logs every X steps."),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default='configs/Sat3DGen_dino_v3.json',
        help="The config of the Vq model to train, leave as None to use standard Vq model configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Path to the prepared VIGOR dataset root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="The resolution used for training images.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100,
        help="Number of logical training epochs. NOTE: internally multiplied by 16 (empirical_epoch_multiplier) to match the released training recipe.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--discr_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--discr_lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=None, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vqgan-training",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ))
    # lambda_opa
    parser.add_argument(
        "--lambda_opa",
        type=float,
        default=30.0,
        help=("The weight of opa loss"))
    # auto resume
    parser.add_argument(
        "--no_auto_resume",
        action="store_true",
        help=("Disable automatic resume from the latest checkpoint under `output_dir`."))
    # generator discriminator training in one iteration
    parser.add_argument(
        "--no_density_regression",
        action="store_true",
        help=("Whether to use density regression loss"))
    parser.add_argument(
        "--lambda_density_regression",
        type=float,
        default=3.5,
        help=("The weight of density regression loss"))
    # no_sat_loss,
    parser.add_argument(
        "--no_sat_loss",
        action="store_true",
        help=("Whether to use satellite loss"))
    # data_txt
    parser.add_argument(
        "--data_txt",
        type=str,
        default=None,
        help="Path to the VIGOR training split file.",
    )
    # depth loss true/false default False
    parser.add_argument(
        "--no_depth_loss",
        action="store_true",
        help=("Whether to use depth loss"))
    parser.add_argument(
        "--lambda_depth_loss",
        type=float,
        default=0.1,
        help=("The weight of depth loss"))
    parser.add_argument(
        "--lambda_pano_recon",
        type=float,
        default=3,
        help=("The weight of street perceptual loss"))
    parser.add_argument(
        "--lambda_sat_recon",
        type=float,
        default=2,
        help=("The weight of satellite perceptual loss"))
    parser.add_argument(
        "--lambda_gen_loss",
        type=float,
        default=0.2,
        help=("The weight of generator loss"))
    parser.add_argument(
        "--density_loose_alpha",
        type=float,
        default=1.0,
        help=("The alpha of density regularisation loose constraint"))
    # sat loss original default false
    # original_no_per_path default is None, type is str
    parser.add_argument(
        "--original_no_per_path",
        type=str,
        default=None,
        help=("The path of original no per training path"))

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_data_dir is None:
        raise ValueError("`--train_data_dir` must point to the prepared VIGOR dataset root.")

    if args.data_txt is None:
        args.data_txt = os.path.join(args.train_data_dir, "train__corrected_all_3city_remove_building.txt")

    return args


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    args = parse_args()

    # Enable TF32 on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=10800)  # change timeout to avoid a strange NCCL bug

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[init_handler],
    )
    accelerator.print(args.output_dir)

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    num_device =  accelerator.num_processes

    if accelerator.is_main_process:
        accelerator.print(f"num_device: {num_device}")
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    generator = Sat3DGen


    if args.model_config_name_or_path is None and args.pretrained_model_name_or_path is None:
        assert False, "You need to specify either a model configuration or a pretrained model."
    elif args.pretrained_model_name_or_path is not None:
        model = generator.from_pretrained(args.pretrained_model_name_or_path)
    else:
        config = generator.load_config(args.model_config_name_or_path)
        model = generator.from_config(config)
    if args.use_ema:
        ema_model = EMAModel(model.parameters(), model_cls=generator, model_config=model.config)

    render_size = model.opt['render_size']
    sr_factor_sign = model.sr_factor 
    patch_size = model.unet_model.patch_size if hasattr(model.unet_model, 'patch_size') else 16
    discriminator = Discriminator(img_resolution=render_size, in_channels=3)
    if not args.no_sat_loss:
        discriminator_sat = Discriminator(img_resolution=render_size//sr_factor_sign)
    dis_pers = Discriminator(img_resolution=render_size, in_channels=3)

    if args.original_no_per_path is not None:
        model = generator.from_pretrained(args.original_no_per_path, subfolder="vqmodel")
        discriminator = Discriminator.from_pretrained(args.original_no_per_path, subfolder="discriminator")
        discriminator_sat = Discriminator.from_pretrained(args.original_no_per_path, subfolder="discriminator_sat") if not args.no_sat_loss else None
        accelerator.print("load original no per path", args.original_no_per_path)


    lpips_loss = lpips.LPIPS('vgg').to(accelerator.device)



    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    model_list = ['vqmodel', 'discriminator', 'dis_pers']
    if not args.no_sat_loss:
        model_list.append('discriminator_sat')

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "vqmodel_ema"))
                for i,name in enumerate(model_list):
                    models[i].save_pretrained(os.path.join(output_dir, name))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vqmodel_ema"),generator)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            for name in reversed(model_list):
                if name == 'vqmodel':
                    model = models.pop()
                    load_model = generator.from_pretrained(input_dir, subfolder="vqmodel")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif name == 'discriminator':
                    discriminator = models.pop()
                    load_model = Discriminator.from_pretrained(input_dir, subfolder="discriminator")
                    discriminator.register_to_config(**load_model.config)
                    discriminator.load_state_dict(load_model.state_dict())
                    del load_model

                elif name == 'discriminator_sat':
                    discriminator_sat = models.pop()
                    load_model = Discriminator.from_pretrained(input_dir, subfolder="discriminator_sat")
                    discriminator_sat.register_to_config(**load_model.config)
                    discriminator_sat.load_state_dict(load_model.state_dict())
                    del load_model
                elif name == 'dis_pers':
                    dis_pers = models.pop()
                    load_model = Discriminator.from_pretrained(input_dir, subfolder="dis_pers")
                    dis_pers.register_to_config(**load_model.config)
                    dis_pers.load_state_dict(load_model.state_dict())
                    del load_model

                else:
                    raise ValueError(f"Unknown model name: {name}")
                    

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    if args.scale_lr:
        lr_scale = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        args.learning_rate *= lr_scale
        args.discr_learning_rate *= lr_scale

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    dis_params = []
    dis_params += list(discriminator.parameters())
    dis_params += list(discriminator_sat.parameters()) if not args.no_sat_loss else []
    dis_params += list(dis_pers.parameters())
    discr_optimizer = optimizer_cls(
        dis_params,
        lr=args.discr_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    ##################################
    # DATLOADER and LR-SCHEDULER     #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    # args.train_batch_size * accelerator.num_processes
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    train_dataset = vigor_dataset(
        render_size,
        sr_factor_sign,
        args.train_data_dir,
        is_train=True,
        data_txt=args.data_txt,
        input_resize=patch_size * 16,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=args.lr_warmup_steps,
    )
    discr_lr_scheduler = get_scheduler(
        args.discr_lr_scheduler,
        optimizer=discr_optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=args.lr_warmup_steps,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    prepare_name = ['model','optimizer','discr_optimizer','lr_scheduler','discr_lr_scheduler','train_dataloader']
    prepare_list = [model,optimizer,discr_optimizer,lr_scheduler,discr_lr_scheduler,train_dataloader]
    for name in model_list[1:]:
        prepare_list.append(eval(name))
        prepare_name.append(name)
    output = accelerator.prepare(*prepare_list)
    for i,name in enumerate(prepare_name):
        if name == 'model':
            model = output[i]
        elif name == 'optimizer':
            optimizer = output[i]
        elif name == 'discr_optimizer':
            discr_optimizer = output[i]
        elif name == 'lr_scheduler':
            lr_scheduler = output[i]
        elif name == 'discr_lr_scheduler':
            discr_lr_scheduler = output[i]
        elif name == 'train_dataloader':
            train_dataloader = output[i]
        elif name == 'discriminator':
            discriminator = output[i]
        elif name == 'discriminator_sat':
            discriminator_sat = output[i]
        elif name == 'dis_pers':
            dis_pers = output[i]

    
    if args.use_ema:
        ema_model.to(accelerator.device)
    # Train!
    # Preserve the original empirical schedule used in the released training recipe.
    empirical_epoch_multiplier = 16
    args.num_train_epochs = args.num_train_epochs * empirical_epoch_multiplier
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Empirical epoch multiplier = {empirical_epoch_multiplier}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    accelerator.print(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}")
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint or (not args.no_auto_resume):
        if resume_from_checkpoint is not None:
            path = find_last_checkpoint(resume_from_checkpoint)
        else:
            path = find_last_checkpoint(args.output_dir)

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            accelerator.wait_for_everyone()
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        ncols=130
    )
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    if not args.no_depth_loss:
        depth_loss_function = MidasLoss(alpha=2).cuda()


    avg_gen_loss, avg_discr_loss = None, None
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        discriminator.train()
        if not args.no_sat_loss:
            discriminator_sat.train()
        dis_pers.train()
        for i, batch in enumerate(train_dataloader):
            sat_gt = batch['sat_img']
            illu_fea = batch['illu_fea']

            str_img_gt_full = batch['str_img']
            gt_sky_mask = batch['sky_mask']
            input_sat = batch['sat_img_input']
            batch_size = input_sat.size(0)

            if 'per_img' in batch: 
                gt_per_img = batch['per_img']

            data_time_m.update(time.time() - end)
            generator_step = True
            generator_step = ((global_step // args.gradient_accumulation_steps) % 2) == 0
            # Train Step
            # The behavior of accelerator.accumulate is to
            # 1. Check if gradients are synced(reached gradient-accumulation_steps)
            # 2. If so sync gradients by stopping the not syncing process
            if generator_step:
                model.train()
                discriminator.requires_grad_(False)
                discriminator.eval()
                if not args.no_sat_loss:
                    discriminator_sat.requires_grad_(False)
                    discriminator_sat.eval()
                dis_pers.requires_grad_(False)
                dis_pers.eval()
            else:
                model.eval()
                discriminator.requires_grad_(True)
                discriminator.train()
                if not args.no_sat_loss:
                    discriminator_sat.requires_grad_(True)
                    discriminator_sat.train()
                dis_pers.requires_grad_(True)
                dis_pers.train()
                
            all_coordinates = None
            if generator_step and not args.no_density_regression:

                initial_coordinates = torch.rand((batch_size, 1000, 3)).to(accelerator.device)
                initial_coordinates_x_y = initial_coordinates[:,:,:2]
                initial_coordinates_z = initial_coordinates[:,:,2:] # lower in position


                perturbed_coordinates_z = initial_coordinates_z + torch.rand_like(
                    initial_coordinates_z) * 0.004 # higher
                perturbed_coordinates = torch.cat([initial_coordinates_x_y, perturbed_coordinates_z],dim=-1)
                all_coordinates = torch.cat(
                    [initial_coordinates, perturbed_coordinates], dim=1)

                all_coordinates = all_coordinates * 2 - 1
            


            if generator_step:
                output = model(input_sat,
                                z_ill=illu_fea, 
                                syn_sat=True if not args.no_sat_loss else False, 
                                syn_pano=True,
                                syn_per=True,
                                intrinsics=batch['intrinsics'],
                                c2w = batch['c2w'],
                                coordinates=all_coordinates
                                )
                pred_sat_opa = output.sat_output.alpha_raw
                pred_sat_depth = output.sat_output.image_depth
                if not args.no_sat_loss:
                    pred_sat_img = output.sat_output.feature_raw[:,:3].mul(2).add(-1).clone()

                    pred_str_idx = output.sat_output.idx
                    gt_sat_img = sat_gt[:,:,pred_str_idx[0]:pred_str_idx[0]+(render_size//sr_factor_sign),pred_str_idx[1]:pred_str_idx[1]+(render_size//sr_factor_sign)].detach()
                    if not args.no_depth_loss:
                        gt_sat_depth = batch['sat_dep']
                        gt_sat_depth = gt_sat_depth[:,:,pred_str_idx[0]:pred_str_idx[0]+(render_size//sr_factor_sign),pred_str_idx[1]:pred_str_idx[1]+(render_size//sr_factor_sign)].detach()
                pred_str_img = output.str_output.sr_image.mul(2).add(-1)
                pred_str_opa = output.str_output.alpha_raw
                pred_sky_img = output.str_output.sky_img[:,:3].mul(2).add(-1).clone()

                pred_per_img = output.per_output.sr_image.mul(2).add(-1)
                pred_per_str_str_opa = output.per_output.alpha_raw
                pred_per_str_sky_img = output.per_output.sky_img[:,:3].clone().mul(2).add(-1)
                
       

                # for dis
                start_h = random.randint(0,(render_size//sr_factor_sign)-1)
                start_w = random.randint(0,(render_size//sr_factor_sign)-1)
                gt_sat_img_dis = sat_gt[:,:,start_h:start_h+(render_size//sr_factor_sign),start_w:start_w+(render_size//sr_factor_sign)].detach()

                gt_str_opa = (1 - gt_sky_mask).detach()
                gt_str_opa = F.interpolate(gt_str_opa, scale_factor=0.5, mode='nearest')




            
                # reconstruction loss
                if not args.no_sat_loss:
                    loss_sat = F.l1_loss(pred_sat_img, gt_sat_img).mean()+ 0.1*lpips_loss(pred_sat_img, gt_sat_img).mean()
                else:
                    loss_sat = 0

                # str opa loss
                opa_loss = 0
                flag = batch['str_img_flag'].squeeze(1).bool()
                opa_loss_batch = (F.binary_cross_entropy_with_logits(pred_str_opa, gt_str_opa,reduction='none'))[flag]
                if opa_loss_batch.numel() > 0: 
                    opa_loss +=  opa_loss_batch.mean()

                gt_sat_opa = torch.ones_like(pred_sat_opa).detach().to(accelerator.device)
                opa_sat_loss = F.binary_cross_entropy_with_logits(pred_sat_opa, gt_sat_opa).mean()
                opa_loss += 0.1 * opa_sat_loss

                # Sky reconstruction loss
                sky_l1_loss = mask_l1_loss(pred_sky_img, F.interpolate(str_img_gt_full, scale_factor=0.5, mode='bilinear',antialias=True), 1-gt_str_opa)
                sky_mean_avg_loss = Custom_non_sky_MeanLoss(output.str_output.sky_img[:,:3], F.interpolate(str_img_gt_full.mul(0.5).add(0.5), scale_factor=0.5, mode='bilinear',antialias=True), 1-gt_str_opa)

                # Panorama reconstruction loss (LPIPS only)
                pano_recon_loss = lpips_loss(pred_str_img, str_img_gt_full).mean()
                # Raw-view auxiliary reconstruction loss (always enabled)
                pred_str_img_low = output.str_output.image_raw_compo[:,:3].clone().mul(2).add(-1)
                gt_for_raw_str = F.interpolate(pred_str_img, scale_factor=0.5, mode='bilinear',antialias=True)
                pano_recon_loss += 0.1 * (0.618*F.l1_loss(pred_str_img_low, gt_for_raw_str).mean()+0.382*lpips_loss(pred_str_img_low, gt_for_raw_str).mean())

                # Perspective reconstruction loss (LPIPS only)
                per_recon_loss = lpips_loss(pred_per_img, gt_per_img).mean()
                pred_per_img_raw = output.per_output.image_raw_compo[:,:3].clone().mul(2).add(-1)
                # Raw-view auxiliary reconstruction loss for perspective (always enabled)
                pred_per_img_raw_gt = F.interpolate(pred_per_img, scale_factor=0.5, mode='bilinear',antialias=True)
                per_recon_loss += 0.1 * (0.618*F.l1_loss(pred_per_img_raw, pred_per_img_raw_gt).mean()+0.382*lpips_loss(pred_per_img_raw, pred_per_img_raw_gt).mean())

                # Generator adversarial loss
                gen_loss_str = F.softplus(-discriminator(pred_str_img)).mean()
                gen_loss_sat = F.softplus(-discriminator_sat(pred_sat_img)).mean() if not args.no_sat_loss else 0
                gen_loss_per = F.softplus(-dis_pers(pred_per_img)).mean()
                gen_loss = gen_loss_sat + gen_loss_str + gen_loss_per

                # Total loss
                loss = args.lambda_sat_recon * loss_sat + args.lambda_pano_recon * pano_recon_loss + args.lambda_opa*opa_loss + sky_l1_loss + 0.1*sky_mean_avg_loss + args.lambda_pano_recon * 0.5 * per_recon_loss
                loss += args.lambda_gen_loss * gen_loss


                if not args.no_depth_loss:
                    depth_loss = compute_depth_loss(depth_loss_function, pred_sat_depth, gt_sat_depth)
                    loss += args.lambda_depth_loss * depth_loss

                if all_coordinates is not None:
                    TVloss = compute_density_tv_loss(output.density, args.density_loose_alpha)
                    loss += args.lambda_density_regression * TVloss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_total_loss = accelerator.gather(loss.repeat(args.train_batch_size)).float().mean()
                # gen_loss
                avg_gen_loss = accelerator.gather(gen_loss.repeat(args.train_batch_size)).float().mean()
                accelerator.backward(loss)
                

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not generator_step:
                # Discriminator loss
                str_img_gt_dis_cat = str_img_gt_full.requires_grad_().clone()
                pred_str_for_dis = pred_str_img.detach().clone()
                real = discriminator(str_img_gt_dis_cat)
                fake = discriminator(pred_str_for_dis)
                loss_str_dis = F.softplus(fake).mean() + F.softplus(-real).mean()
                loss_str_dis += gradient_penalty(str_img_gt_dis_cat, real)

                gt_per_dis = gt_per_img.requires_grad_().clone()
                pred_per = pred_per_img.detach().clone()
                real_per = dis_pers(gt_per_dis)
                fake_per = dis_pers(pred_per)
                loss_per_dis = F.softplus(fake_per).mean() + F.softplus(-real_per).mean()
                loss_per_dis += gradient_penalty(gt_per_dis, real_per)
                loss_sat_dis = 0
                if not args.no_sat_loss:
                    pred_sat_img = pred_sat_img.detach().clone()
                    gt_sat_img_dis = gt_sat_img_dis.requires_grad_().clone()

                    real_sat = discriminator_sat(gt_sat_img_dis)
                    fake_sat = discriminator_sat(pred_sat_img)
                    loss_sat_dis = F.softplus(fake_sat).mean() + F.softplus(-real_sat).mean()
                    gp_sat = gradient_penalty(gt_sat_img_dis, real_sat)
                    loss_sat_dis += gp_sat

                loss_dis = loss_sat_dis + loss_str_dis + loss_per_dis
                accelerator.backward(loss_dis)
                avg_discr_loss = accelerator.gather(loss_dis.repeat(args.train_batch_size)).mean()

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                    accelerator.clip_grad_norm_(dis_pers.parameters(), args.max_grad_norm)
                    if not args.no_sat_loss:
                        accelerator.clip_grad_norm_(discriminator_sat.parameters(), args.max_grad_norm)

                discr_optimizer.step()
                discr_lr_scheduler.step()
                discr_optimizer.zero_grad()

            batch_time_m.update(time.time() - end)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
            global_step += 1

            if accelerator.sync_gradients and accelerator.is_main_process:
                interval = 301
                if (global_step) % interval == 0: 
                    # add gen_loss
                    outdict = {'G_l': f'{avg_total_loss:.3f}',"o_l": f'{opa_loss:.3f}',"g_l": f'{gen_loss:.3f}'}
                    if 'density' in output.keys():
                        outdict["TVloss"] = f'{TVloss:.6f}'
                    progress_bar.set_postfix(outdict,True)
                    progress_bar.update(interval) 
                # wait for both generator and discriminator to settle
                # Log metrics
                if global_step % args.log_steps == 0:
                    samples_per_second_per_gpu = (
                        args.gradient_accumulation_steps * args.train_batch_size / batch_time_m.val
                    )
                    logs = {
                        "step_discr_loss": avg_discr_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    if avg_total_loss is not None:
                        logs["step_total_loss"] = avg_total_loss.item()
                    if avg_gen_loss is not None:
                        logs["step_gen_loss"] = avg_gen_loss.item()
                    if opa_loss:
                        logs["opa_loss"] = opa_loss.item()
                    if sky_l1_loss:
                        logs["sky_l1_loss"] = sky_l1_loss.item()
                    if pano_recon_loss:
                        logs["pano_recon_loss"] = pano_recon_loss.item()
                    if per_recon_loss:
                        logs["per_recon_loss"] = per_recon_loss.item()
                    if loss_sat:
                        logs["loss_sat"] = loss_sat.item()
                    if loss_sat_dis:
                        logs["loss_sat_dis"] = loss_sat_dis.item()
                    if loss_str_dis:
                        logs["loss_str_dis"] = loss_str_dis.item()
                    if loss_per_dis:
                        logs["loss_per_dis"] = loss_per_dis.item()
                    if not args.no_depth_loss:
                        logs["depth_loss"] = depth_loss.item()
                    if all_coordinates is not None:
                        logs["density_regression"] = TVloss.item()
                        logs['density_max'] = output.density.max().item()
                    #gen_loss_sat
                    if gen_loss_sat:
                        logs["gen_loss_sat"] = gen_loss_sat.item()
                    if gen_loss_str:
                        logs["gen_loss_str"] = gen_loss_str.item()
                    if gen_loss_per:
                        logs["gen_loss_per"] = gen_loss_per.item()
                    if sky_mean_avg_loss:
                        logs["sky_mean_avg_loss"] = sky_mean_avg_loss.item()
                    accelerator.log(logs, step=global_step)

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()
                # Save model checkpoint
                if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps - 1:
                    save_checkpoint(accelerator, args, global_step, logger)
            end = time.time()
            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
