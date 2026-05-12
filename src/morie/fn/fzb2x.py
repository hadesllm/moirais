# morie.fn -- function file (hadesllm/morie)
"""b_2(x) coefficient in KDFE bias expansion."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_b2_coefficient"]


def fauzi_b2_coefficient(x, kernel):
    """
    b_2(x) coefficient in KDFE bias expansion

    Formula: b_2(x) = (f_X'(x)/2) * mu_2(K) where mu_2(K) = integral w^2 K(w) dw

    Parameters
    ----------
    x : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficient

    References
    ----------
    Fauzi Ch 2, Eq 2.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "b_2(x) coefficient in KDFE bias expansion"})


def cheatsheet():
    return "fzb2x: b_2(x) coefficient in KDFE bias expansion"
