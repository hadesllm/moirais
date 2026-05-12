r"""Bertscore f1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_bertscore_f1"]


def kamath_ch8_bertscore_f1(P_BERT, R_BERT):
    r"""
    Bertscore f1.

    Formula: F_{BERT} = 2\cdot\frac{P_{BERT}\cdot R_{BERT}}{P_{BERT}+R_{BERT}}

    Parameters
    ----------
    P_BERT : array-like
        Input data.
    R_BERT : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.9, p. 325
    r"""
    P_BERT = np.atleast_1d(np.asarray(P_BERT, dtype=float))
    n = len(P_BERT)
    result = float(np.mean(P_BERT))
    se = float(np.std(P_BERT, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bertscore f1."})


def cheatsheet():
    return "km121: Bertscore f1."
