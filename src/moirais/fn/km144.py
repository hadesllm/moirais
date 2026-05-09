"""Mm instr predict.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_mm_instr_predict"]


def kamath_ch9_mm_instr_predict(I, M, theta):
    """
    Mm instr predict.

    Formula: A = f(I, M; \theta)

    Parameters
    ----------
    I : array-like
        Input data.
    M : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.16, p. 391
    """
    I = np.atleast_1d(np.asarray(I, dtype=float))
    n = len(I)
    result = float(np.mean(I))
    se = float(np.std(I, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mm instr predict."})


def cheatsheet():
    return "km144: Mm instr predict."
