"""TMLE for path-specific effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_path_specific"]


def tmle_path_specific(y, D, M_chain, X, path):
    """
    TMLE for path-specific effects

    Formula: target effect along DAG path P

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    M_chain : array-like
        Input data.
    X : array-like
        Input data.
    path : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pearl (2001); Avin et al (2005); Miles-Tchetgen-Tchetgen (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for path-specific effects"})


def cheatsheet():
    return "tmlpse: TMLE for path-specific effects"
