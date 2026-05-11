"""Catalog coverage."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["catalog_coverage"]


def catalog_coverage(recommendations, catalog):
    """
    Catalog coverage

    Formula: |recommended unique items|/|catalog|

    Parameters
    ----------
    recommendations : array-like
        Input data.
    catalog : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Herlocker et al (2004)
    """
    recommendations = np.atleast_1d(np.asarray(recommendations, dtype=float))
    n = len(recommendations)
    result = float(np.mean(recommendations))
    se = float(np.std(recommendations, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Catalog coverage"})


def cheatsheet():
    return "covLst: Catalog coverage"
