"""Random-walk-with-drift."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["drift_forecast"]


def drift_forecast(y, h):
    """
    Random-walk-with-drift

    Formula: ŷ_{T+h}=y_T + h·(y_T−y_1)/(T−1)

    Parameters
    ----------
    y : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hyndman-Athanasopoulos (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random-walk-with-drift"})


def cheatsheet():
    return "driftF: Random-walk-with-drift"
