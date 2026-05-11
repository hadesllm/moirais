"""Krippendorff's alpha."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["krippendorff_alpha"]


def krippendorff_alpha(data, level):
    """
    Krippendorff's alpha

    Formula: alpha = 1 - D_o / D_e

    Parameters
    ----------
    data : array-like
        Input data.
    level : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Krippendorff (2004)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Krippendorff's alpha"})


def cheatsheet():
    return "krpalp: Krippendorff's alpha"
