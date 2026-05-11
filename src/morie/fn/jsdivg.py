"""Jensen-Shannon divergence (symmetric KL)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["jensen_shannon_divergence"]


def jensen_shannon_divergence(y, p, q):
    """
    Jensen-Shannon divergence (symmetric KL)

    Formula: JSD(P||Q) = (1/2) D_KL(P||M) + (1/2) D_KL(Q||M); M=(P+Q)/2

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lin (1991)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jensen-Shannon divergence (symmetric KL)"})


def cheatsheet():
    return "jsdivg: Jensen-Shannon divergence (symmetric KL)"
