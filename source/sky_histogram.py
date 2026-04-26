"""Shared utility for computing sky-colour histograms from a panorama image."""

import numpy as np
import torch


def compute_sky_histogram(sky_image_np: np.ndarray,
                          hist_range: tuple = (-1, 1),
                          bins: int = 100,
                          skip_bins: int = 10) -> np.ndarray:
    """Compute a normalised colour histogram from a masked sky image.

    Parameters
    ----------
    sky_image_np : np.ndarray
        Sky image array of shape ``(C, H, W)`` (single image, no batch dim).
        Pixel values should lie within *hist_range*.
    hist_range : tuple
        ``(min, max)`` passed to ``np.histogram``.
    bins : int
        Number of histogram bins.
    skip_bins : int
        Number of leading bins to discard (removes near-zero / masked pixels).

    Returns
    -------
    np.ndarray
        Concatenated normalised histogram of shape ``((bins - skip_bins) * C,)``.
    """
    channel_histograms = []
    for channel in sky_image_np:
        histo = np.histogram(channel.flatten(), bins=bins, range=hist_range)[0]
        histo = histo[skip_bins:]
        if histo.sum() != 0:
            histo = histo / histo.sum()
        channel_histograms.append(histo)
    return np.concatenate(channel_histograms)
