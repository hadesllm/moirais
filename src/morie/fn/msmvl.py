# morie.fn — function file (hadesllm/morie)
"""Elastic deformation field computation. 'I'm Ms. Marvel!' -- Kamala Khan"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def elastic_deformation(
    shape: tuple[int, int],
    *,
    alpha: float = 10.0,
    sigma: float = 3.0,
    seed: int | None = None,
) -> DescriptiveResult:
    """Generate a random elastic deformation field (Simard et al., 2003).

    Creates smooth random displacement fields by convolving uniform noise
    with a Gaussian kernel, then scaling by *alpha*.  Used in data
    augmentation for image classification.

    Parameters
    ----------
    shape : tuple of (H, W)
        Spatial dimensions of the deformation field.
    alpha : float
        Deformation intensity.
    sigma : float
        Gaussian kernel standard deviation (smoothness).
    seed : int or None
        RNG seed.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``dx`` (H x W), ``dy`` (H x W),
        ``magnitude`` (H x W), ``mean_displacement``, ``max_displacement``.
    """
    H, W = shape
    if H < 3 or W < 3:
        raise ValueError("shape dimensions must be >= 3")

    rng = np.random.default_rng(seed)
    dx = rng.uniform(-1, 1, (H, W))
    dy = rng.uniform(-1, 1, (H, W))

    k_size = max(3, int(6 * sigma) | 1)
    ax = np.arange(-k_size // 2 + 1, k_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    from scipy.signal import convolve2d

    dx = alpha * convolve2d(dx, kernel, mode="same", boundary="symm")
    dy = alpha * convolve2d(dy, kernel, mode="same", boundary="symm")

    mag = np.sqrt(dx**2 + dy**2)

    return DescriptiveResult(
        name="elastic_deformation",
        value={
            "dx": dx,
            "dy": dy,
            "magnitude": mag,
            "mean_displacement": float(mag.mean()),
            "max_displacement": float(mag.max()),
        },
        extra={"shape": shape, "alpha": alpha, "sigma": sigma},
    )


msmvl = elastic_deformation


def cheatsheet() -> str:
    return "elastic_deformation({}) -> Elastic deformation field computation. 'I'm Ms. Marvel!' -- "
