"""Bottom-up hierarchy aggregation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bottom_up_aggregation"]


def bottom_up_aggregation(bottoms, S):
    """
    Bottom-up hierarchy aggregation

    Formula: ŷ_aggregate = sum bottom forecasts

    Parameters
    ----------
    bottoms : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hyndman-Athanasopoulos (2018) §11
    """
    bottoms = np.atleast_1d(np.asarray(bottoms, dtype=float))
    n = len(bottoms)
    result = float(np.mean(bottoms))
    se = float(np.std(bottoms, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bottom-up hierarchy aggregation"})


def cheatsheet():
    return "botUp: Bottom-up hierarchy aggregation"
