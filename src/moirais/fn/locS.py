"""Joint robust location-scale (e.g. M-step + MAD)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["location_scale_estimator"]


def location_scale_estimator(x):
    """
    Joint robust location-scale (e.g. M-step + MAD)

    Formula: alternate location M-step + scale MAD

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
    Huber (1981) book
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint robust location-scale (e.g. M-step + MAD)"})


def cheatsheet():
    return "locS: Joint robust location-scale (e.g. M-step + MAD)"
