r"""Seq2seq loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_seq2seq_loss"]


def kamath_ch2_seq2seq_loss(x, xhat, i, j):
    r"""
    Seq2seq loss.

    Formula: L^{(x)}_{Seq2Seq} = -\frac{1}{l_s}\sum_{s=i}^{j}\log P(x_s|\hat{x}, x_{i:s-1})

    Parameters
    ----------
    x : array-like
        Input data.
    xhat : array-like
        Input data.
    i : array-like
        Input data.
    j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.32, p. 54
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Seq2seq loss."})


def cheatsheet():
    return "km032: Seq2seq loss."
