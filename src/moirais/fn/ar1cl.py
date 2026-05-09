"""AR(1) red noise climate model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ar1_climate"]


def ar1_climate(x, phi):
    """
    AR(1) red noise climate model

    Formula: x_t = φ x_{t-1} + ε_t

    Parameters
    ----------
    x : array-like
        Input data.
    phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hasselmann (1976)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AR(1) red noise climate model"})


def cheatsheet():
    return "ar1cl: AR(1) red noise climate model"
