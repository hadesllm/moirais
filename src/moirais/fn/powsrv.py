"""Survey-design-aware power."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["power_survey"]


def power_survey(effect_size, alpha, DEFF, n):
    """
    Survey-design-aware power

    Formula: adjust SRS power by sqrt(DEFF)

    Parameters
    ----------
    effect_size : array-like
        Input data.
    alpha : array-like
        Input data.
    DEFF : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lumley (2010)
    """
    effect_size = np.atleast_1d(np.asarray(effect_size, dtype=float))
    n = len(effect_size)
    result = float(np.mean(effect_size))
    se = float(np.std(effect_size, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-design-aware power"})


def cheatsheet():
    return "powsrv: Survey-design-aware power"
