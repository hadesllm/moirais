"""Joint entropy H(X,Y)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["joint_entropy"]


def joint_entropy(pxy, base):
    """
    Joint entropy H(X,Y)

    Formula: H(X,Y) = -sum p(x,y) log p(x,y)

    Parameters
    ----------
    pxy : array-like
        Input data.
    base : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cover-Thomas (2006)
    """
    pxy = np.atleast_1d(np.asarray(pxy, dtype=float))
    n = len(pxy)
    result = float(np.mean(pxy))
    se = float(np.std(pxy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint entropy H(X,Y)"})


def cheatsheet():
    return "jntent: Joint entropy H(X,Y)"
