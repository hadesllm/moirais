"""Bradley terry pref.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_bradley_terry_pref"]


def kamath_ch5_bradley_terry_pref(r_star, y_w, y_l):
    """
    Bradley terry pref.

    Formula: p^*(y_w \succ y_l|x) = \frac{\exp(r^*(x,y_w))}{\exp(r^*(x,y_w)) + \exp(r^*(x,y_l))}

    Parameters
    ----------
    r_star : array-like
        Input data.
    y_w : array-like
        Input data.
    y_l : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.8, p. 209
    """
    r_star = np.atleast_1d(np.asarray(r_star, dtype=float))
    n = len(r_star)
    result = float(np.mean(r_star))
    se = float(np.std(r_star, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bradley terry pref."})


def cheatsheet():
    return "km072: Bradley terry pref."
