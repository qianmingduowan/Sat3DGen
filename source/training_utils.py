import os
import shutil


def save_checkpoint(accelerator, args, global_step, logger):
    """Save model checkpoint and enforce checkpoints_total_limit."""
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")

    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                checkpoint_path = os.path.join(args.output_dir, removing_checkpoint)
                if os.path.exists(checkpoint_path):
                    try:
                        shutil.rmtree(checkpoint_path)
                        logger.info(f'Directory "{checkpoint_path}" has been removed.')
                    except Exception as e:
                        logger.info(f'Error removing directory "{checkpoint_path}": {e}. Continuing with the next item.')
                else:
                    logger.info(f'Directory "{checkpoint_path}" does not exist.')
