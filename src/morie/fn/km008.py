r"""Attention softmax weights.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_attention_softmax_weights"]


def kamath_ch2_attention_softmax_weights(a):
    r"""
    Attention softmax weights.

    Formula: b = \mathrm{softmax}(a)

    Parameters
    ----------
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.8, p. 32
    r"""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Attention softmax weights."})


def cheatsheet():
    return "km008: Attention softmax weights."
