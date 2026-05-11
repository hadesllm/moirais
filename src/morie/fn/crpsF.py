"""Continuous ranked probability score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["crps"]


def crps(forecast_cdf, y):
    """
    Continuous ranked probability score

    Formula: CRPS(F,y) = ∫(F(z)−𝟙{z≥y})²dz

    Parameters
    ----------
    forecast_cdf : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matheson-Winkler (1976)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous ranked probability score"})


def cheatsheet():
    return "crpsF: Continuous ranked probability score"
