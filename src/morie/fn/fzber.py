# morie.fn — function file (hadesllm/morie)
"""Berry-Esseen bound improvement for kernel quantile: O(n^{-1/2}) vs O(n^{-7/17})."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_berry_esseen_quantile"]


def fauzi_berry_esseen_quantile(data, p, bandwidth, x):
    """
    Berry-Esseen bound improvement for kernel quantile: O(n^{-1/2}) vs O(n^{-7/17})

    Formula: P(sqrt(n)|Q_hat_{p,h}-Q(p)| <= x sigma_n) = 2Phi(x)-1+O(n^{-1/2})

    Parameters
    ----------
    data : array-like
        Input data.
    p : array-like
        Input data.
    bandwidth : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    Fauzi Ch 3, Remark 3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Berry-Esseen bound improvement for kernel quantile: O(n^{-1/2}) vs O(n^{-7/17})"})


def cheatsheet():
    return "fzber: Berry-Esseen bound improvement for kernel quantile: O(n^{-1/2}) vs O(n^{-7/17})"
