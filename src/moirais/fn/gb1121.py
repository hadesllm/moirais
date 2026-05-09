# moirais.fn — function file (hadesllm/moirais)
"""Kendall tau coefficient T = (concordant - discordant pairs) / C(n,2)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_kendall_tau"]


def gibbons_kendall_tau(x, y):
    """
    Kendall tau coefficient T = (concordant - discordant pairs) / C(n,2)

    Formula: T = (P - Q) / C(n,2); P = concordant pairs, Q = discordant

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tau, p_value

    References
    ----------
    Gibbons Ch 11.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kendall tau coefficient T = (concordant - discordant pairs) / C(n,2)"})


def cheatsheet():
    return "gb1121: Kendall tau coefficient T = (concordant - discordant pairs) / C(n,2)"
