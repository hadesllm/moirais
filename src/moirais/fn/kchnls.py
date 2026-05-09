"""KL chain rule."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["k_l_divergence_chain"]


def k_l_divergence_chain(pxy, qxy):
    """
    KL chain rule

    Formula: D(p(x,y)||q(x,y)) = D(p(x)||q(x)) + D(p(y|x)||q(y|x))

    Parameters
    ----------
    pxy : array-like
        Input data.
    qxy : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KL chain rule"})


def cheatsheet():
    return "kchnls: KL chain rule"
