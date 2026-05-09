# moirais.fn — function file (hadesllm/moirais)
"""Kernel-based non-stationary spatial prediction (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import cdist


def knspl(
    coords: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    """
    Predict at target locations using kernel-weighted local regression.

    A Nadaraya-Watson estimator with spatially adaptive weights provides
    non-stationary spatial prediction (local polynomial smoothing).

    :param coords: Training coordinates (n, 2).
    :param values: Training values (n,).
    :param target: Prediction coordinates (m, 2).
    :param bandwidth: Kernel bandwidth; defaults to median NN distance.
    :param kernel: ``'gaussian'`` or ``'epanechnikov'``.
    :return: dict with ``predictions``, ``weights``, ``bandwidth``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 8.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if target.shape[1] != 2:
        raise ValueError(f"target must have 2 columns, got {target.shape[1]}.")

    if bandwidth is None:
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        bandwidth = float(np.median(d.min(axis=1)))
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    dists = cdist(target, coords)
    if kernel == "gaussian":
        W = np.exp(-0.5 * (dists / bandwidth) ** 2)
    elif kernel == "epanechnikov":
        u = dists / bandwidth
        W = np.where(u <= 1, 0.75 * (1 - u ** 2), 0.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums
    predictions = W_norm @ values

    return {
        "predictions": predictions,
        "weights": W_norm,
        "bandwidth": bandwidth,
        "n_train": n,
        "n_target": len(target),
    }


knspl_fn = knspl


def cheatsheet() -> str:
    return "knspl({}) -> Kernel-based non-stationary spatial prediction."
