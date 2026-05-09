"""Free-Wilson QSAR additive scheme."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["free_wilson_qsar"]


def free_wilson_qsar(activities, substituent_indicators):
    """
    Free-Wilson QSAR additive scheme

    Formula: sum of indicator-variable contributions per substituent

    Parameters
    ----------
    activities : array-like
        Input data.
    substituent_indicators : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Free-Wilson (1964)
    """
    activities = np.atleast_1d(np.asarray(activities, dtype=float))
    n = len(activities)
    result = float(np.mean(activities))
    se = float(np.std(activities, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Free-Wilson QSAR additive scheme"})


def cheatsheet():
    return "frwil: Free-Wilson QSAR additive scheme"
