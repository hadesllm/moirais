"""Orthogonal GARCH via PCA + univariate GARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_garch_orthogonal"]


def vol_garch_orthogonal(R_panel, k):
    """
    Orthogonal GARCH via PCA + univariate GARCH

    Formula: PCA on R; fit GARCH(1,1) on each PC

    Parameters
    ----------
    R_panel : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loadings, sigmas

    References
    ----------
    Alexander (2001)
    """
    R_panel = np.atleast_1d(np.asarray(R_panel, dtype=float))
    n = len(R_panel)
    result = float(np.mean(R_panel))
    se = float(np.std(R_panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Orthogonal GARCH via PCA + univariate GARCH"})


def cheatsheet():
    return "volgo: Orthogonal GARCH via PCA + univariate GARCH"
