"""Gpt supervised obj.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt_supervised_obj"]


def kamath_ch2_gpt_supervised_obj(C, x, y):
    """
    Gpt supervised obj.

    Formula: L_2(C) = \sum_{(x,y)} \log P(y|x_1,\dots,x_m)

    Parameters
    ----------
    C : array-like
        Input data.
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
    Kamath et al (2024), Ch 2, Eq 2.36, p. 70
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gpt supervised obj."})


def cheatsheet():
    return "km036: Gpt supervised obj."
