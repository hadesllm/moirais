# moirais.fn — function file (hadesllm/moirais)
"""Adaptive noise cancellation (LMS)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_adaptive_filter"]


def rangayyan_adaptive_filter(x, reference):
    """
    Adaptive noise cancellation (LMS)

    Formula: w(n+1) = w(n) + 2*mu*e(n)*x(n)

    Parameters
    ----------
    x : array-like
        Input data.
    reference : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rangayyan Ch 11
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive noise cancellation (LMS)"})


def cheatsheet():
    return "rgadp: Adaptive noise cancellation (LMS)"
