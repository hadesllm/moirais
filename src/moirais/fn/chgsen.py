"""Change-of-variance sensitivity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["change_of_variance"]


def change_of_variance(IF):
    """
    Change-of-variance sensitivity

    Formula: how V(T) responds to ε contamination

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
    Hampel et al (1986)
    """
    IF = np.atleast_1d(np.asarray(IF, dtype=float))
    n = len(IF)
    result = float(np.mean(IF))
    se = float(np.std(IF, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Change-of-variance sensitivity"})


def cheatsheet():
    return "chgsen: Change-of-variance sensitivity"
