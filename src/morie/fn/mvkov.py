# morie.fn — function file (hadesllm/morie)
"""Moving window variogram (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def mvkov(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    window_radius: float | None = None,
    n_lags: int = 10,
) -> dict:
    """
    Compute moving-window variograms for non-stationary diagnostics.

    A local empirical variogram is computed within a moving window
    centred at each observation, revealing how spatial dependence
    varies across the study region.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param window_radius: Radius of the moving window; defaults to
        half the maximum pairwise distance.
    :param n_lags: Number of lag bins for each local variogram.
    :return: dict with ``local_variograms`` (list of dicts per location),
        ``window_radius``, ``n_lags``.
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

    dist_mat = squareform(pdist(coords))
    if window_radius is None:
        window_radius = float(dist_mat.max() / 2.0)
    if window_radius <= 0:
        raise ValueError("window_radius must be positive.")

    local_variograms = []
    for i in range(n):
        in_window = np.where(dist_mat[i] <= window_radius)[0]
        if len(in_window) < 3:
            local_variograms.append({"lags": [], "semivariance": []})
            continue
        local_dists = dist_mat[np.ix_(in_window, in_window)]
        local_vals = values[in_window]
        max_d = local_dists.max()
        if max_d == 0:
            local_variograms.append({"lags": [], "semivariance": []})
            continue
        lag_edges = np.linspace(0, max_d, n_lags + 1)
        lags = []
        sv = []
        for k in range(n_lags):
            mask = (local_dists > lag_edges[k]) & (local_dists <= lag_edges[k + 1])
            pairs = np.where(mask)
            if len(pairs[0]) == 0:
                continue
            gamma = 0.5 * np.mean((local_vals[pairs[0]] - local_vals[pairs[1]]) ** 2)
            lags.append(float((lag_edges[k] + lag_edges[k + 1]) / 2))
            sv.append(float(gamma))
        local_variograms.append({"lags": lags, "semivariance": sv})

    return {
        "local_variograms": local_variograms,
        "window_radius": window_radius,
        "n_lags": n_lags,
        "n": n,
    }


mvkov_fn = mvkov


def cheatsheet() -> str:
    return "mvkov({}) -> Moving window variogram for non-stationarity diagnostics."
