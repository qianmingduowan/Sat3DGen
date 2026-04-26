import torch


def compute_density_tv_loss(density, alpha):
    """Compute total variation loss on density to regularise geometry.

    The density tensor is expected to contain initial and perturbed samples
    concatenated along dim-1 (i.e. ``density.shape[1] == 2 * N``).  The loss
    encourages the perturbed density not to exceed the initial density by more
    than *alpha*, which acts as a soft margin.

    Args:
        density: Tensor of shape ``(B, 2*N, ...)``.
        alpha: Scalar margin added to the initial density.

    Returns:
        Scalar loss value.
    """
    density_len = density.shape[1] // 2
    density_initial = density[:, :density_len]
    density_perturbed = density[:, density_len:]
    return torch.mean(torch.relu(density_perturbed - (density_initial + alpha)))
