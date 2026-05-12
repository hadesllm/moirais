"""Anisotropy detection by comparison of directional variograms."""
import numpy as np
from scipy import stats as _scistats
from ._richresult import RichResult

__all__ = ["anisotropy_test"]


def _directional_pairs(coords, angle_rad, tol_rad):
    """Return pair indices whose connecting vector falls within tol of angle."""
    n = coords.shape[0]
    iu = np.triu_indices(n, k=1)
    dv = coords[iu[1]] - coords[iu[0]]
    if dv.shape[1] == 1:
        return iu  # 1D: everyone is along the only direction
    ang = np.arctan2(dv[:, 1], dv[:, 0])
    # Map to [0, pi) (variogram is symmetric)
    ang = np.mod(ang, np.pi)
    target = np.mod(angle_rad, np.pi)
    diff = np.abs(ang - target)
    diff = np.minimum(diff, np.pi - diff)
    mask = diff <= tol_rad
    return (iu[0][mask], iu[1][mask])


def anisotropy_test(x, coords, n_dirs: int = 4, tol_deg: float = 22.5):
    """
    Geometric-anisotropy test by comparing directional semivariances
    at a single representative lag (median of inter-point distances).

    A Levene-style F-test on the squared-difference distributions across
    `n_dirs` evenly spaced directions in [0, pi).

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, 2)
    n_dirs : int
        Number of directional sectors (default 4 -> 0, 45, 90, 135 deg).
    tol_deg : float
        Half-width of each angular sector (degrees).

    Returns
    -------
    RichResult with payload:  statistic (F), p_value, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = x.size
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match x ({n})")
    if coords.shape[1] == 1:
        # Anisotropy is undefined in 1D -- return null result
        return RichResult(payload={
            "statistic": 0.0, "p_value": 1.0, "n": int(n),
            "method": "Anisotropy test (1D: trivially isotropic)",
        })
    tol_rad = np.deg2rad(tol_deg)
    angles = np.linspace(0.0, np.pi, n_dirs, endpoint=False)

    groups = []
    means = []
    for ang in angles:
        i, j = _directional_pairs(coords, ang, tol_rad)
        if len(i) < 2:
            continue
        diffs2 = (x[i] - x[j]) ** 2
        groups.append(diffs2)
        means.append(0.5 * diffs2.mean())
    if len(groups) < 2:
        return RichResult(payload={
            "statistic": float("nan"), "p_value": float("nan"),
            "n": int(n), "method": "Anisotropy test (insufficient pairs)",
        })
    # Levene's test on squared-differences -- robust to non-normality
    stat, p = _scistats.levene(*groups, center="median")
    return RichResult(payload={
        "statistic": float(stat),
        "p_value": float(p),
        "directional_gamma": [float(m) for m in means],
        "directions_deg": [float(np.rad2deg(a)) for a in angles[:len(means)]],
        "n": int(n),
        "method": f"Anisotropy test (Levene across {n_dirs} directions)",
    })


def cheatsheet():
    return "aniso: Anisotropy test (directional variogram comparison)"


# CANONICAL TEST
# 2D grid:  coords = [(i,j) for i in 0..3 for j in 0..3], 16 points
# Isotropic x: x = i+j  -> Levene p should be > 0.05 (fail to reject)
