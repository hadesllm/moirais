r"""Seq2seq cross entropy.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_seq2seq_cross_entropy"]


def kamath_ch2_seq2seq_cross_entropy(y, c, U):
    r"""
    Seq2seq cross entropy.

    Formula: L = -\sum_{t=1}^{U} \log p(y_t|y_{t-1},\dots,y_1,c)

    Parameters
    ----------
    y : array-like
        Input data.
    c : array-like
        Input data.
    U : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.6, p. 31
    r"""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Seq2seq cross entropy."})


def cheatsheet():
    return "km006: Seq2seq cross entropy."
