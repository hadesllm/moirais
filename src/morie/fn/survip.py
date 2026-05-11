"""Survey-corrected p-value."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_p_value"]


def survey_p_value(test_stat, DEFF):
    """
    Survey-corrected p-value

    Formula: adjust p-value via design-correction factor

    Parameters
    ----------
    test_stat : array-like
        Input data.
    DEFF : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Korn-Graubard (1999)
    """
    test_stat = np.atleast_1d(np.asarray(test_stat, dtype=float))
    n = len(test_stat)
    result = float(np.mean(test_stat))
    se = float(np.std(test_stat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-corrected p-value"})


def cheatsheet():
    return "survip: Survey-corrected p-value"
