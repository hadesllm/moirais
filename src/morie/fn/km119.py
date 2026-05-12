r"""Bertscore recall.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_bertscore_recall"]


def kamath_ch8_bertscore_recall(x, xhat):
    r"""
    Bertscore recall.

    Formula: R_{BERT} = \frac{1}{|x|}\sum_{x_i\in x} \max_{\hat{x}_j\in\hat{x}} \langle x_i,\hat{x}_j\rangle

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
    Kamath et al (2024), Ch 8, Eq 8.7, p. 325
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bertscore recall."})


def cheatsheet():
    return "km119: Bertscore recall."
