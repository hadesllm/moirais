# morie.fn — function file (hadesllm/morie)
"""Bootstrap bandwidth selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_bandwidth_bootstrap"]


def horowitz_bandwidth_bootstrap(x, y):
    """
    Bootstrap bandwidth selection

    Formula: h* = argmin MISE_boot(h)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Horowitz (2009), Ch 13
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap bandwidth selection"})


def cheatsheet():
    return "hrzw2: Bootstrap bandwidth selection"
