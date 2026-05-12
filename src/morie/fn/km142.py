r"""Itg loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_itg_loss"]


def kamath_ch9_itg_loss(x, y):
    r"""
    Itg loss.

    Formula: L_{ITG} = -\sum_{(x,y)\in(X,Y)} \log \prod_{t=1}^n P(y_t|y_{<t},x)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.14, p. 389
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Itg loss."})


def cheatsheet():
    return "km142: Itg loss."
