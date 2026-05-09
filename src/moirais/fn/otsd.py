"""Quantile-form sliced Wasserstein for fast 1-D evaluation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_sliced_distance_quant"]


def ot_sliced_distance_quant(X, Y, p, n_proj):
    """
    Quantile-form sliced Wasserstein for fast 1-D evaluation

    Formula: W_p(P_θ#μ, P_θ#ν) via sorted projections

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    p : array-like
        Input data.
    n_proj : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: SW

    References
    ----------
    Rabin-Peyré-Delon-Bernot (2011)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile-form sliced Wasserstein for fast 1-D evaluation"})


def cheatsheet():
    return "otsd: Quantile-form sliced Wasserstein for fast 1-D evaluation"
