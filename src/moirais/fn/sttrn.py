"""Spatio-temporal trend surface (Schabenberger & Gotway Ch 9)."""

import numpy as np


def sttrn(
    coords: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    *,
    spatial_order: int = 1,
    temporal_order: int = 1,
    interaction: bool = True,
) -> dict:
    """
    Fit a spatio-temporal polynomial trend surface.

    Constructs a design matrix with polynomial terms in space and
    time (and optionally interactions) and fits by OLS.

    :param coords: Observation coordinates (n, 2).
    :param times: Observation times (n,).
    :param values: Observed values (n,).
    :param spatial_order: Maximum polynomial order in space.
    :param temporal_order: Maximum polynomial order in time.
    :param interaction: Include space-time interaction terms.
    :return: dict with ``coefficients``, ``fitted``, ``residuals``, ``r_squared``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 9.
    """
    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if len(times) != n:
        raise ValueError("times must have same length as values.")

    x, y = coords[:, 0], coords[:, 1]
    cols = [np.ones(n)]

    for p in range(1, spatial_order + 1):
        for q in range(p + 1):
            cols.append(x ** (p - q) * y ** q)

    for p in range(1, temporal_order + 1):
        cols.append(times ** p)

    if interaction:
        for sp in range(1, spatial_order + 1):
            for tp in range(1, temporal_order + 1):
                cols.append(x ** sp * times ** tp)
                cols.append(y ** sp * times ** tp)

    X = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
    fitted = X @ coeffs
    residuals = values - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {
        "coefficients": coeffs,
        "fitted": fitted,
        "residuals": residuals,
        "r_squared": float(r2),
        "n_terms": len(coeffs),
        "n": n,
    }


sttrn_fn = sttrn


def cheatsheet() -> str:
    return "sttrn({}) -> Spatio-temporal polynomial trend surface."
