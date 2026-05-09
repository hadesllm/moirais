# moirais.fn — function file (hadesllm/moirais)
"""Deformation kernel / space warping (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import cdist


def dkern(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_basis: int = 5,
    seed: int | None = None,
) -> dict:
    """
    Spatial deformation via thin-plate spline basis (Sampson & Guttorp 1992).

    Maps geographic coordinates to a deformed space where stationarity
    holds approximately, using radial basis functions.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param n_basis: Number of basis knots.
    :param seed: Random seed for knot selection.
    :return: dict with ``deformed_coords``, ``knots``, ``coefficients``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Sampson, P. D. & Guttorp, P. (1992). Nonparametric estimation of
    nonstationary spatial covariance structure. *JASA*, 87(417), 108-119.

    Schabenberger & Gotway (2005), Ch. 8.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_basis, n), replace=False)
    knots = coords[idx]

    dists_to_knots = cdist(coords, knots)
    r2 = dists_to_knots ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(r2 > 0, r2 * np.log(r2), 0.0)

    ones = np.ones((n, 1))
    A = np.hstack([ones, coords, phi])
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)

    deformed = phi @ coeffs[3:, None] if len(coeffs) > 3 else np.zeros((n, 1))
    deformed_coords = coords + np.column_stack([deformed.ravel(), deformed.ravel()])

    return {
        "deformed_coords": deformed_coords,
        "knots": knots,
        "coefficients": coeffs,
        "n_basis": len(knots),
        "n": n,
    }


dkern_fn = dkern


def cheatsheet() -> str:
    return "dkern({}) -> Deformation kernel / space warping."
