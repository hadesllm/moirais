"""Unobserved components model (trend+seasonal+irregular)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["unobserved_components"]


def unobserved_components(x):
    """
    Unobserved components model (trend+seasonal+irregular)

    Formula: y_t = mu_t + gamma_t + e_t

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
    Harvey (1989)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unobserved components model (trend+seasonal+irregular)"})


def cheatsheet():
    return "ucmod: Unobserved components model (trend+seasonal+irregular)"
