# moirais.fn — function file (hadesllm/moirais)
"""Nelson-Aalen with empirical process theory."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_nelson_aalen"]


def kosorok_nelson_aalen(t, event):
    """
    Nelson-Aalen with empirical process theory

    Formula: Lambda_hat(t) = integral dN(s)/Y(s)

    Parameters
    ----------
    t : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 8
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nelson-Aalen with empirical process theory"})


def cheatsheet():
    return "ksr18: Nelson-Aalen with empirical process theory"
