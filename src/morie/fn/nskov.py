# morie.fn -- function file (hadesllm/morie)
"""Non-stationary kernel covariance estimation (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import cdist


def nskov(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    """
    Estimate non-stationary covariance using kernel smoothing.

    At each location the local covariance is estimated with a kernel
    weighted average, allowing the covariance structure to vary across
    the study region.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param bandwidth: Kernel bandwidth. If *None*, uses median pairwise distance.
    :param kernel: ``'gaussian'`` or ``'bisquare'``.
    :return: dict with ``cov_matrix``, ``bandwidth``, ``kernel``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, Ch. 8.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")

    dists = cdist(coords, coords)
    if bandwidth is None:
        bandwidth = float(np.median(dists[dists > 0]))
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    if kernel == "gaussian":
        W = np.exp(-0.5 * (dists / bandwidth) ** 2)
    elif kernel == "bisquare":
        W = (1.0 - (dists / bandwidth) ** 2) ** 2
        W[dists > bandwidth] = 0.0
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    z = values - values.mean()
    outer = np.outer(z, z)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums
    cov_matrix = W_norm @ outer

    return {
        "cov_matrix": cov_matrix,
        "bandwidth": bandwidth,
        "kernel": kernel,
        "n": n,
    }


nskov_fn = nskov


def cheatsheet() -> str:
    return "nskov({}) -> Non-stationary kernel covariance estimation."
