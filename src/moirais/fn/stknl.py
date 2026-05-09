"""Spatio-temporal kernel smoothing (Schabenberger & Gotway Ch 9)."""

import numpy as np
from scipy.spatial.distance import cdist


def stknl(
    coords: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    target_coords: np.ndarray,
    target_times: np.ndarray,
    *,
    spatial_bw: float | None = None,
    temporal_bw: float | None = None,
) -> dict:
    """
    Spatio-temporal Nadaraya-Watson kernel smoother.

    Estimates the mean at target space-time locations using a product
    kernel with separate spatial and temporal bandwidths.

    :param coords: Training coordinates (n, 2).
    :param times: Training times (n,).
    :param values: Training values (n,).
    :param target_coords: Prediction coordinates (m, 2).
    :param target_times: Prediction times (m,).
    :param spatial_bw: Spatial bandwidth; defaults to median NN distance.
    :param temporal_bw: Temporal bandwidth; defaults to median time gap.
    :return: dict with ``predictions``, ``spatial_bw``, ``temporal_bw``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 9.
    """
    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    n = len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if len(times) != n:
        raise ValueError("times must have same length as values.")
    if target_coords.ndim == 1:
        target_coords = target_coords.reshape(1, -1)
    m = len(target_coords)
    if len(target_times) != m:
        raise ValueError("target_times must match target_coords length.")

    if spatial_bw is None:
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        spatial_bw = float(np.median(d.min(axis=1)))
    if temporal_bw is None:
        t_sorted = np.sort(times)
        gaps = np.diff(t_sorted)
        temporal_bw = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0

    sdists = cdist(target_coords, coords)
    tdists = np.abs(target_times[:, None] - times[None, :])

    Ws = np.exp(-0.5 * (sdists / spatial_bw) ** 2)
    Wt = np.exp(-0.5 * (tdists / temporal_bw) ** 2)
    W = Ws * Wt

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    predictions = (W / row_sums) @ values

    return {
        "predictions": predictions,
        "spatial_bw": spatial_bw,
        "temporal_bw": temporal_bw,
        "n_train": n,
        "n_target": m,
    }


stknl_fn = stknl


def cheatsheet() -> str:
    return "stknl({}) -> Spatio-temporal kernel smoothing."
