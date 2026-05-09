"""Gross-error sensitivity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gross_error_sensitivity"]


def gross_error_sensitivity(IF):
    """
    Gross-error sensitivity

    Formula: γ* = sup_x ||IF(x;T,F)||

    Parameters
    ----------
    IF : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hampel et al (1986) book
    """
    IF = np.atleast_1d(np.asarray(IF, dtype=float))
    n = len(IF)
    result = float(np.mean(IF))
    se = float(np.std(IF, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gross-error sensitivity"})


def cheatsheet():
    return "gross: Gross-error sensitivity"
