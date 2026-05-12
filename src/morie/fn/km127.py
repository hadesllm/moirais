r"""Geval score.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_geval_score"]


def kamath_ch8_geval_score(s_i, p):
    r"""
    Geval score.

    Formula: \mathrm{score} = \sum_{i=1}^n p(s_i)\cdot s_i

    Parameters
    ----------
    s_i : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.15, p. 328
    r"""
    s_i = np.atleast_1d(np.asarray(s_i, dtype=float))
    n = len(s_i)
    result = float(np.mean(s_i))
    se = float(np.std(s_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Geval score."})


def cheatsheet():
    return "km127: Geval score."
