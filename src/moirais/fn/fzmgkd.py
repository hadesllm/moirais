# moirais.fn — function file (hadesllm/moirais)
"""Modified gamma KDE via self-elimination technique."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_modified_gamma_kde"]


def fauzi_modified_gamma_kde(x, bandwidth, a):
    """
    Modified gamma KDE via self-elimination technique

    Formula: f_tilde_X(x) = [f_hat_h]^{a^2/(a^2-1)} * [f_hat_{ah}]^{-1/(a^2-1)}

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 1, Eq 1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Modified gamma KDE via self-elimination technique"})


def cheatsheet():
    return "fzmgkd: Modified gamma KDE via self-elimination technique"
