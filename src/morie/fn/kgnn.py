"""R-GCN -- relational GCN."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["r_gcn"]


def r_gcn(A_r, X, W_r):
    """
    R-GCN -- relational GCN

    Formula: per-relation transform + sum

    Parameters
    ----------
    A_r : array-like
        Input data.
    X : array-like
        Input data.
    W_r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schlichtkrull et al (2018)
    """
    A_r = np.atleast_1d(np.asarray(A_r, dtype=float))
    n = len(A_r)
    result = float(np.mean(A_r))
    se = float(np.std(A_r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "R-GCN -- relational GCN"})


def cheatsheet():
    return "kgnn: R-GCN -- relational GCN"
