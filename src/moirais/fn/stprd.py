"""Spatio-temporal prediction intervals (Schabenberger & Gotway Ch 9)."""

import numpy as np
from scipy.spatial.distance import cdist


def stprd(
    coords: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    target_coord: np.ndarray,
    target_time: float,
    *,
    spatial_range: float = 1.0,
    temporal_range: float = 1.0,
    sill: float = 1.0,
    nugget: float = 0.0,
    alpha: float = 0.05,
) -> dict:
    """
    Compute spatio-temporal kriging prediction with intervals.

    Uses simple kriging with a separable exponential covariance to
    produce prediction and interval at a target space-time point.

    :param coords: Training coordinates (n, 2).
    :param times: Training times (n,).
    :param values: Training values (n,).
    :param target_coord: Target coordinate (2,).
    :param target_time: Target time.
    :param spatial_range: Spatial range.
    :param temporal_range: Temporal range.
    :param sill: Covariance sill.
    :param nugget: Nugget variance.
    :param alpha: Significance level for interval.
    :return: dict with ``prediction``, ``variance``, ``lower``, ``upper``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 9.
    """
    from scipy.stats import norm

    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    target_coord = np.asarray(target_coord, dtype=float).ravel()
    n = len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if len(times) != n:
        raise ValueError("times length must match values.")

    sdists = cdist(coords, target_coord.reshape(1, -1)).ravel()
    tdists = np.abs(times - target_time)

    Cs = sill * np.exp(-sdists / (spatial_range + 1e-12))
    Ct = np.exp(-tdists / (temporal_range + 1e-12))
    c0 = Cs * Ct

    sdist_mat = cdist(coords, coords)
    tdist_mat = np.abs(times[:, None] - times[None, :])
    C = sill * np.exp(-sdist_mat / (spatial_range + 1e-12)) * np.exp(-tdist_mat / (temporal_range + 1e-12))
    np.fill_diagonal(C, C.diagonal() + nugget)

    try:
        w = np.linalg.solve(C, c0)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(C, c0, rcond=None)[0]

    pred = float(w @ values)
    var = max(float(sill - w @ c0), 0.0)
    z_crit = norm.ppf(1 - alpha / 2)
    se = np.sqrt(var)

    return {
        "prediction": pred,
        "variance": var,
        "std_error": float(se),
        "lower": pred - z_crit * se,
        "upper": pred + z_crit * se,
        "alpha": alpha,
        "n": n,
    }


stprd_fn = stprd


def cheatsheet() -> str:
    return "stprd({}) -> Spatio-temporal prediction with intervals."
