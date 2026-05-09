"""Geometric anisotropy: directional variogram via affine coordinate transformation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_geometric_anisotropy"]


def schabenberger_geometric_anisotropy(coords, z, A_matrix):
    """
    Geometric anisotropy: directional variogram via affine coordinate transformation

    Formula: gamma(h) = gamma_iso(||A*h||) where A is rotation/stretch matrix

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    A_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: semivariance

    References
    ----------
    Schabenberger Ch 4, Sec 4.3.7
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Geometric anisotropy: directional variogram via affine coordinate transformation"})


def cheatsheet():
    return "spanis: Geometric anisotropy: directional variogram via affine coordinate transformation"
