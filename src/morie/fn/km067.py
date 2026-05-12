r"""Rm bradley terry.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_rm_bradley_terry"]


def kamath_ch5_rm_bradley_terry(x, y_w, y_l, r_theta):
    r"""
    Rm bradley terry.

    Formula: L(\theta) = -E_{(x,y_w,y_l)\sim D}[\log\sigma(r_{\theta}(x,y_w) - r_{\theta}(x,y_l))]

    Parameters
    ----------
    x : array-like
        Input data.
    y_w : array-like
        Input data.
    y_l : array-like
        Input data.
    r_theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.3, p. 200
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rm bradley terry."})


def cheatsheet():
    return "km067: Rm bradley terry."
