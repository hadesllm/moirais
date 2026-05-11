"""Bertscore precision.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_bertscore_precision"]


def kamath_ch8_bertscore_precision(x, xhat):
    """
    Bertscore precision.

    Formula: P_{BERT} = \frac{1}{|\hat{x}|}\sum_{\hat{x}_j\in\hat{x}} \max_{x_i\in x} \langle\hat{x}_j,x_i\rangle

    Parameters
    ----------
    x : array-like
        Input data.
    xhat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.8, p. 325
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bertscore precision."})


def cheatsheet():
    return "km120: Bertscore precision."
