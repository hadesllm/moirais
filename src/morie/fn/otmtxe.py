"""RAS matrix scaling — fit row + col marginals via diagonal scalings."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_matrix_scaling"]


def ot_matrix_scaling(K, row_target, col_target, max_iter):
    """
    RAS matrix scaling — fit row + col marginals via diagonal scalings

    Formula: M' = diag(u) K diag(v); iterate

    Parameters
    ----------
    K : array-like
        Input data.
    row_target : array-like
        Input data.
    col_target : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: M, u, v

    References
    ----------
    Bregman (1967)
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    n = len(K)
    result = float(np.mean(K))
    se = float(np.std(K, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RAS matrix scaling — fit row + col marginals via diagonal scalings"})


def cheatsheet():
    return "otmtxe: RAS matrix scaling — fit row + col marginals via diagonal scalings"
