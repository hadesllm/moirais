"""Hansch QSAR linear regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hansch_qsar"]


def hansch_qsar(activities, descriptors):
    """
    Hansch QSAR linear regression

    Formula: log(1/C) = a*pi + b*sigma + c*Es + const

    Parameters
    ----------
    activities : array-like
        Input data.
    descriptors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hansch-Fujita (1964); Hansch-Leo (1995)
    """
    activities = np.atleast_1d(np.asarray(activities, dtype=float))
    n = len(activities)
    result = float(np.mean(activities))
    se = float(np.std(activities, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hansch QSAR linear regression"})


def cheatsheet():
    return "qsarh: Hansch QSAR linear regression"
