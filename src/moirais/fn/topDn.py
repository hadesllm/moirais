"""Top-down disaggregation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["top_down"]


def top_down(top, props):
    """
    Top-down disaggregation

    Formula: split top forecast by historical proportions

    Parameters
    ----------
    top : array-like
        Input data.
    props : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hyndman-Athanasopoulos (2018) §11
    """
    top = np.atleast_1d(np.asarray(top, dtype=float))
    n = len(top)
    result = float(np.mean(top))
    se = float(np.std(top, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Top-down disaggregation"})


def cheatsheet():
    return "topDn: Top-down disaggregation"
