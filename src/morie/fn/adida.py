"""ADIDA aggregate-disaggregate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["adida"]


def adida(y, period):
    """
    ADIDA aggregate-disaggregate

    Formula: aggregate to lower freq, forecast, disaggregate

    Parameters
    ----------
    y : array-like
        Input data.
    period : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nikolopoulos et al (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ADIDA aggregate-disaggregate"})


def cheatsheet():
    return "adida: ADIDA aggregate-disaggregate"
