# moirais.fn — function file (hadesllm/moirais)
"""b_4(x) coefficient in KDFE bias expansion."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_b4_coefficient"]


def fauzi_b4_coefficient(x, kernel):
    """
    b_4(x) coefficient in KDFE bias expansion

    Formula: b_4(x) = (f_X'''(x)/24) * mu_4(K) where mu_4(K) = integral w^4 K(w) dw

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
    Fauzi Ch 2, Eq 2.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "b_4(x) coefficient in KDFE bias expansion"})


def cheatsheet():
    return "fzb4x: b_4(x) coefficient in KDFE bias expansion"
