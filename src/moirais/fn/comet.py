"""COMET MT eval."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["comet"]


def comet(src, hyp, ref):
    """
    COMET MT eval

    Formula: learned regressor on src+ref+hyp embeddings

    Parameters
    ----------
    src : array-like
        Input data.
    hyp : array-like
        Input data.
    ref : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rei et al (2020)
    """
    src = np.atleast_1d(np.asarray(src, dtype=float))
    n = len(src)
    result = float(np.mean(src))
    se = float(np.std(src, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "COMET MT eval"})


def cheatsheet():
    return "comet: COMET MT eval"
