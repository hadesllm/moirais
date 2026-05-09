"""Spatio-temporal IDE model (Schabenberger & Gotway Ch 9)."""

import numpy as np
from scipy.spatial.distance import cdist


def stide(
    coords: np.ndarray,
    data: np.ndarray,
    *,
    bandwidth: float | None = None,
    n_steps: int | None = None,
) -> dict:
    """
    Fit a spatio-temporal integro-difference equation (IDE) model.

    .. math::

        z_{t+1}(s) = \\int k(s, u) z_t(u) du + \\varepsilon_{t+1}(s)

    The redistribution kernel *k* is estimated from data using a
    Gaussian kernel with the given bandwidth.

    :param coords: Spatial coordinates (n, 2).
    :param data: Spatio-temporal observations (T, n).
    :param bandwidth: Kernel bandwidth; defaults to median pairwise distance.
    :param n_steps: Number of forward prediction steps; defaults to T-1.
    :return: dict with ``kernel_matrix``, ``predictions``, ``residuals``,
        ``bandwidth``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Wikle, C. K. (2002). A kernel-based spectral model for non-Gaussian
    spatio-temporal processes. *Statistical Modelling*, 2(4), 299-314.

    Schabenberger & Gotway (2005), Ch. 9.
    """
    coords = np.asarray(coords, dtype=float)
    data = np.asarray(data, dtype=float)
    T, n = data.shape
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if T < 2:
        raise ValueError("Need at least 2 time steps.")

    dists = cdist(coords, coords)
    if bandwidth is None:
        bandwidth = float(np.median(dists[dists > 0]))
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    K = np.exp(-0.5 * (dists / bandwidth) ** 2)
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    K = K / row_sums

    if n_steps is None:
        n_steps = T - 1

    predictions = np.zeros((n_steps, n))
    predictions[0] = K @ data[0]
    for t in range(1, n_steps):
        predictions[t] = K @ predictions[t - 1]

    residuals = data[1 : n_steps + 1] - predictions[: min(n_steps, T - 1)]

    return {
        "kernel_matrix": K,
        "predictions": predictions,
        "residuals": residuals,
        "bandwidth": bandwidth,
        "T": T,
        "n": n,
    }


stide_fn = stide


def cheatsheet() -> str:
    return "stide({}) -> Spatio-temporal integro-difference equation model."
