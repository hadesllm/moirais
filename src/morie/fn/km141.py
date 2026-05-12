r"""Itm loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_itm_loss"]


def kamath_ch9_itm_loss(theta, v, t, y):
    r"""
    Itm loss.

    Formula: L_{ITM}(\theta) = -E_{(v,t)}[y\log s_{\theta}(v,t) + (1-y)\log(1-s_{\theta}(v,t))]

    Parameters
    ----------
    theta : array-like
        Input data.
    v : array-like
        Input data.
    t : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.13, p. 388
    r"""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Itm loss."})


def cheatsheet():
    return "km141: Itm loss."
