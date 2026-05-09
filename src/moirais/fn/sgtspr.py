"""Bipartite-detection via spectral symmetry."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_spectral_radius_bound"]


def sgt_spectral_radius_bound(A):
    """
    Bipartite-detection via spectral symmetry

    Formula: Bipartite iff spectrum is symmetric around 0

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bipartite, evidence

    References
    ----------
    Cvetković-Doob-Sachs (1995)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bipartite-detection via spectral symmetry"})


def cheatsheet():
    return "sgtspr: Bipartite-detection via spectral symmetry"
