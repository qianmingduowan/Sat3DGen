import torch


def compute_depth_loss(depth_loss_function, pred_depth, gt_depth):
    """Compute scale-invariant depth loss for satellite view.

    Args:
        depth_loss_function: A ``MidasLoss`` instance.
        pred_depth: Predicted depth map.
        gt_depth: Ground-truth depth map.

    Returns:
        Scalar loss value (the first element returned by ``MidasLoss``).
    """
    mask = torch.ones_like(gt_depth) > 0
    return depth_loss_function(pred_depth + 1, (gt_depth * 5 + 1), mask)[0]
