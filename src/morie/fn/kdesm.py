# morie.fn — function file (hadesllm/morie)
"""Kernel density estimation (Gaussian kernel)."""

import numpy as np

from ._containers import DescriptiveResult
def kde_smooth(x, bandwidth=None, n_points: int = 256, **kwargs) -> DescriptiveResult:
    """
    Kernel density estimation with a Gaussian kernel.

    Uses Silverman's rule of thumb for bandwidth if not provided:
    h = 0.9 · min(σ, IQR/1.34) · n^(-1/5).

    :param x: array-like of observations.
    :param bandwidth: Kernel bandwidth. If None, uses Silverman's rule.
    :param n_points: Number of evaluation grid points.
    :return: DescriptiveResult with grid and density values.

    References
    ----------
    Silverman BW (1986). Density Estimation for Statistics and
    Data Analysis. Chapman & Hall, London.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    if bandwidth is None:
        std = float(np.std(x, ddof=1))
        iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
        bandwidth = 0.9 * min(std, iqr / 1.34) * n ** (-0.2)
        if bandwidth <= 0:
            bandwidth = 0.1
    grid = np.linspace(float(np.min(x)) - 3 * bandwidth, float(np.max(x)) + 3 * bandwidth, n_points)
    density = np.zeros(n_points)
    for xi in x:
        density += np.exp(-0.5 * ((grid - xi) / bandwidth) ** 2)
    density /= n * bandwidth * np.sqrt(2 * np.pi)
    return DescriptiveResult(
        name="kde_smooth",
        value=float(bandwidth),
        extra={"grid": grid.tolist(), "density": density.tolist(), "bandwidth": float(bandwidth), "n": n},
    )


kdesm = kde_smooth


def cheatsheet() -> str:
    return "kde_smooth({}) -> Kernel density estimation (Gaussian kernel)."
