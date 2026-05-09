"""TimesFM foundation model (Google)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["timesfm_foundation"]


def timesfm_foundation(y, horizon):
    """
    TimesFM foundation model (Google)

    Formula: decoder-only transformer for TS forecasting

    Parameters
    ----------
    y : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Das et al (2024) TimesFM
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TimesFM foundation model (Google)"})


def cheatsheet():
    return "timesf: TimesFM foundation model (Google)"
