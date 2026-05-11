# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Anisotropic variogram model (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def anvgm(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_directions: int = 4,
    n_lags: int = 10,
    tolerance: float = 22.5,
) -> dict:
    """
    Compute directional variograms for anisotropy assessment.

    Variograms are computed along ``n_directions`` equally spaced
    azimuths (0 to 180 degrees) with an angular tolerance, revealing
    directional dependence in the spatial process.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param n_directions: Number of directional bins in [0, 180).
    :param n_lags: Number of lag bins.
    :param tolerance: Angular tolerance in degrees for each direction.
    :return: dict with ``directions`` (degrees), ``lags``, ``semivariance``
        (n_directions x n_lags).
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

    directions = np.linspace(0, 180, n_directions, endpoint=False)
    tol_rad = np.radians(tolerance)

    dist_mat = squareform(pdist(coords))
    max_d = dist_mat.max()
    lag_edges = np.linspace(0, max_d, n_lags + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    angles = np.arctan2(dy, dx)
    angles[angles < 0] += np.pi

    sv = np.full((n_directions, n_lags), np.nan)
    for d_idx, direction in enumerate(directions):
        dir_rad = np.radians(direction)
        angle_diff = np.abs(angles - dir_rad)
        angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
        dir_mask = angle_diff <= tol_rad

        for k in range(n_lags):
            lag_mask = (dist_mat > lag_edges[k]) & (dist_mat <= lag_edges[k + 1])
            mask = dir_mask & lag_mask
            ii, jj = np.where(mask)
            if len(ii) < 2:
                continue
            sv[d_idx, k] = 0.5 * np.mean((values[ii] - values[jj]) ** 2)

    return {
        "directions": directions,
        "lags": lag_centers,
        "semivariance": sv,
        "n_directions": n_directions,
        "tolerance": tolerance,
        "n": n,
    }


anvgm_fn = anvgm


def cheatsheet() -> str:
    return "anvgm({}) -> Anisotropic variogram model."
