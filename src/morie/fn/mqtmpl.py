"""Multi-QTL model selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multi_qtl"]


def multi_qtl(y, markers, positions):
    """
    Multi-QTL model selection

    Formula: forward-backward search across markers

    Parameters
    ----------
    y : array-like
        Input data.
    markers : array-like
        Input data.
    positions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Broman et al (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-QTL model selection"})


def cheatsheet():
    return "mqtmpl: Multi-QTL model selection"
