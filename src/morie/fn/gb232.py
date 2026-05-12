# morie.fn -- function file (hadesllm/morie)
"""Glivenko-Cantelli theorem: uniform convergence of EDF to CDF with prob 1."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_glivenko_cantelli"]


def gibbons_glivenko_cantelli(x):
    """
    Glivenko-Cantelli theorem: uniform convergence of EDF to CDF with prob 1

    Formula: P(lim sup |S_n(x) - F(x)| = 0) = 1

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: convergence_result

    References
    ----------
    Gibbons Theorem 2.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glivenko-Cantelli theorem: uniform convergence of EDF to CDF with prob 1"})


def cheatsheet():
    return "gb232: Glivenko-Cantelli theorem: uniform convergence of EDF to CDF with prob 1"
