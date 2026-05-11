"""Agreement scores between legislators."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_agreement"]


def spatial_agreement(x):
    """
    Agreement scores between legislators

    Formula: A_ij = proportion of same votes

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Armstrong Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Agreement scores between legislators"})


def cheatsheet():
    return "sptag: Agreement scores between legislators"
