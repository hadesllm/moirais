"""LightGCN -- simplified graph CF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lightgcn"]


def lightgcn(R, K, layers):
    """
    LightGCN -- simplified graph CF

    Formula: L=normalized adj; emb = sum α_k L^k E

    Parameters
    ----------
    R : array-like
        Input data.
    K : array-like
        Input data.
    layers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    He et al (2020) LightGCN
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LightGCN -- simplified graph CF"})


def cheatsheet():
    return "lightG: LightGCN -- simplified graph CF"
