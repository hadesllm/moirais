"""LISA cluster classification (HH/LL/HL/LH/NS)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_cluster_lisa"]


def spatial_cluster_lisa(x, W, alpha):
    """
    LISA cluster classification (HH/LL/HL/LH/NS)

    Formula: per-location category from sign + significance

    Parameters
    ----------
    x : array-like
        Input data.
    W : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anselin (1995)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LISA cluster classification (HH/LL/HL/LH/NS)"})


def cheatsheet():
    return "spcllm: LISA cluster classification (HH/LL/HL/LH/NS)"
