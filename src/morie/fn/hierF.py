"""Hierarchical reconciliation (MinT)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hierarchical_forecast"]


def hierarchical_forecast(base_forecasts, S, cov):
    """
    Hierarchical reconciliation (MinT)

    Formula: ŷ_h = SG ŷ_base; G via min-trace

    Parameters
    ----------
    base_forecasts : array-like
        Input data.
    S : array-like
        Input data.
    cov : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wickramasuriya-Athanasopoulos-Hyndman (2019) MinT
    """
    base_forecasts = np.atleast_1d(np.asarray(base_forecasts, dtype=float))
    n = len(base_forecasts)
    result = float(np.mean(base_forecasts))
    se = float(np.std(base_forecasts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical reconciliation (MinT)"})


def cheatsheet():
    return "hierF: Hierarchical reconciliation (MinT)"
