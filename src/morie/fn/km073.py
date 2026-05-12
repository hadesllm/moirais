r"""Pref sigmoid form.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_pref_sigmoid_form"]


def kamath_ch5_pref_sigmoid_form(r_star):
    r"""
    Pref sigmoid form.

    Formula: p^*(y_w \succ y_l|x) = \sigma(r^*(x,y_w) - r^*(x,y_l))

    Parameters
    ----------
    r_star : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.9, p. 210
    r"""
    r_star = np.atleast_1d(np.asarray(r_star, dtype=float))
    n = len(r_star)
    result = float(np.mean(r_star))
    se = float(np.std(r_star, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pref sigmoid form."})


def cheatsheet():
    return "km073: Pref sigmoid form."
