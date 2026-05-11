"""Years lived with disability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["yld_calculation"]


def yld_calculation(prevalence, disability, duration):
    """
    Years lived with disability

    Formula: YLD = prevalence × disability_weight × duration

    Parameters
    ----------
    prevalence : array-like
        Input data.
    disability : array-like
        Input data.
    duration : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    WHO GBD
    """
    prevalence = np.atleast_1d(np.asarray(prevalence, dtype=float))
    n = len(prevalence)
    result = float(np.mean(prevalence))
    se = float(np.std(prevalence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Years lived with disability"})


def cheatsheet():
    return "yldyear: Years lived with disability"
