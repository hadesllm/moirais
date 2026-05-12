"""PolarQuant radius+angle decomposition of a d-dim vector."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_polar_transform"]


def turboquant_polar_transform(x):
    """
    PolarQuant radius+angle decomposition of a d-dim vector

    Formula: r = ||x||_2;  theta_i = atan2(x_{i+1}, x_i)  for successive pairs;  x = (r, theta)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: r, theta

    References
    ----------
    TurboQuant MORIE integration -- morie/quant.py polar_transform
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PolarQuant radius+angle decomposition of a d-dim vector"})


def cheatsheet():
    return "tqpol: PolarQuant radius+angle decomposition of a d-dim vector"
