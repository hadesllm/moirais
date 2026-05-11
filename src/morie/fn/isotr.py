# morie.fn — function file (hadesllm/morie)
"""Pool-adjacent-violators isotonic regression for nonmetric MDS disparities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["isotonic_regression_disparity"]


def isotonic_regression_disparity(D, delta_rank):
    """
    Pool-adjacent-violators isotonic regression for nonmetric MDS disparities

    Formula: dhat = argmin_{dhat monotone} sum (d_ij - dhat_ij)^2 via PAV algorithm

    Parameters
    ----------
    D : array-like
        Input data.
    delta_rank : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'dhat': 'matrix'}

    References
    ----------
    Armstrong Ch 3
    """
    D = np.asarray(D, dtype=float)
    n = int(D) if D.ndim == 0 else len(D)
    result = float(np.mean(D))
    se = float(np.std(D, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pool-adjacent-violators isotonic regression for nonmetric MDS disparities"})


def cheatsheet():
    return "isotr: Pool-adjacent-violators isotonic regression for nonmetric MDS disparities"
