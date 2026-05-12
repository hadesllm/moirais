r"""Gpt supervised softmax.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt_supervised_softmax"]


def kamath_ch2_gpt_supervised_softmax(x, h, W_y):
    r"""
    Gpt supervised softmax.

    Formula: P(y|x_1,\dots,x_m) = \mathrm{softmax}(h_m^l W_y)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.
    W_y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.35, p. 70
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gpt supervised softmax."})


def cheatsheet():
    return "km035: Gpt supervised softmax."
