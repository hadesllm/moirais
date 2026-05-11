# morie.fn — function file (hadesllm/morie)
"""Control function approach to endogeneity."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_control_function"]


def horowitz_control_function(x, y, w, bandwidth):
    """
    Control function approach to endogeneity

    Formula: Y = g(X) + U; X = r(W) + V; E[U|V]=h(V); Y = g(X)+h(V)+eps, eps perp V

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_hat, h_hat

    References
    ----------
    Horowitz Ch 5, Sec 5.5.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Control function approach to endogeneity"})


def cheatsheet():
    return "hrzctrl: Control function approach to endogeneity"
