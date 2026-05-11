# morie.fn — function file (hadesllm/morie)
"""Local variogram estimation (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import cdist


def lclvg(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    target: np.ndarray | None = None,
    bandwidth: float | None = None,
    n_lags: int = 10,
) -> dict:
    """
    Estimate local variograms at target locations using kernel weights.

    Unlike the global variogram, the local variogram weights each pair
    by proximity to the target location, capturing spatial heterogeneity
    in the dependence structure.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param target: Target location(s) (m, 2); defaults to all coords.
    :param bandwidth: Kernel bandwidth; defaults to median pairwise distance.
    :param n_lags: Number of lag bins.
    :return: dict with ``lags``, ``semivariance`` (m x n_lags), ``bandwidth``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 8.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")

    if target is None:
        target = coords.copy()
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target.reshape(1, -1)

    pair_dists = cdist(coords, coords)
    if bandwidth is None:
        bandwidth = float(np.median(pair_dists[pair_dists > 0]))

    max_lag = pair_dists.max()
    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    target_dists = cdist(target, coords)
    W_target = np.exp(-0.5 * (target_dists / bandwidth) ** 2)

    m = len(target)
    sv_all = np.zeros((m, n_lags))
    for t_idx in range(m):
        w = W_target[t_idx]
        for k in range(n_lags):
            mask = (pair_dists > lag_edges[k]) & (pair_dists <= lag_edges[k + 1])
            ii, jj = np.where(mask)
            if len(ii) == 0:
                continue
            pair_w = w[ii] * w[jj]
            total_w = pair_w.sum()
            if total_w == 0:
                continue
            sq_diff = (values[ii] - values[jj]) ** 2
            sv_all[t_idx, k] = 0.5 * np.sum(pair_w * sq_diff) / total_w

    return {
        "lags": lag_centers,
        "semivariance": sv_all,
        "bandwidth": bandwidth,
        "n_targets": m,
        "n": n,
    }


lclvg_fn = lclvg


def cheatsheet() -> str:
    return "lclvg({}) -> Local variogram estimation with kernel weights."
