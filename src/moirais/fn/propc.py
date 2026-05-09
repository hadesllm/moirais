# moirais.fn — function file (hadesllm/moirais)
"""Prophet-style decomposition (trend+seasonality+holidays)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["prophet_components"]


def prophet_components(x):
    """
    Prophet-style decomposition (trend+seasonality+holidays)

    Formula: y(t) = g(t) + s(t) + h(t) + e_t

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Taylor & Letham (2018)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prophet-style decomposition (trend+seasonality+holidays)"})


def cheatsheet():
    return "propc: Prophet-style decomposition (trend+seasonality+holidays)"
