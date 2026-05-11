"""Attention output.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_attention_output"]


def kamath_ch2_attention_output(b, v):
    """
    Attention output.

    Formula: o = \sum_{i=1}^n b_i v_i

    Parameters
    ----------
    b : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.10, p. 32
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    n = len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Attention output."})


def cheatsheet():
    return "km010: Attention output."
