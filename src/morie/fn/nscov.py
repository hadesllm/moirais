# morie.fn — function file (hadesllm/morie)
"""Non-stationary covariance function (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import cdist


def nscov(
    coords: np.ndarray,
    *,
    sigma_func: str = "linear",
    range_func: str = "linear",
    nugget: float = 0.0,
) -> dict:
    """
    Evaluate a parametric non-stationary covariance function.

    The covariance between locations s_i and s_j depends on
    spatially varying sill and range parameters, following the
    Paciorek & Schervish (2006) construction generalised from
    Schabenberger & Gotway Ch 8.

    :param coords: Locations (n, 2).
    :param sigma_func: How the local sill varies: ``'constant'`` or
        ``'linear'`` (varies with x-coordinate).
    :param range_func: How the local range varies: ``'constant'`` or
        ``'linear'``.
    :param nugget: Nugget variance added to the diagonal.
    :return: dict with ``cov_matrix``, ``local_sill``, ``local_range``.
    :raises ValueError: If coords have wrong shape.

    References
    ----------
    Paciorek, C. J. & Schervish, M. J. (2006). Spatial modelling using
    a new class of nonstationary covariance functions. *Environmetrics*,
    17(5), 483-506.

    Schabenberger & Gotway (2005), Ch. 8.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got {coords.shape}.")

    n = len(coords)
    x = coords[:, 0]
    x_scaled = (x - x.min()) / (x.max() - x.min() + 1e-12)

    if sigma_func == "linear":
        local_sill = 0.5 + x_scaled
    else:
        local_sill = np.ones(n)

    if range_func == "linear":
        local_range = 0.5 + x_scaled
    else:
        local_range = np.ones(n)

    cov = np.zeros((n, n))
    dists = cdist(coords, coords)
    for i in range(n):
        for j in range(i, n):
            si = local_sill[i]
            sj = local_sill[j]
            ri = local_range[i]
            rj = local_range[j]
            avg_r = np.sqrt(ri * rj)
            det_factor = (2 * ri * rj / (ri ** 2 + rj ** 2 + 1e-12))
            c = np.sqrt(si * sj) * det_factor * np.exp(-dists[i, j] ** 2 / (ri ** 2 + rj ** 2 + 1e-12))
            cov[i, j] = c
            cov[j, i] = c

    np.fill_diagonal(cov, cov.diagonal() + nugget)

    return {
        "cov_matrix": cov,
        "local_sill": local_sill,
        "local_range": local_range,
        "nugget": nugget,
        "n": n,
    }


nscov_fn = nscov


def cheatsheet() -> str:
    return "nscov({}) -> Non-stationary covariance function."
