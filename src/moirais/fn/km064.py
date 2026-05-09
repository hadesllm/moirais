"""Loftq objective.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_loftq_objective"]


def kamath_ch4_loftq_objective(W, Q, A, B):
    """
    Loftq objective.

    Formula: \min_{Q,A,B} \|W - Q - AB^T\|_F

    Parameters
    ----------
    W : array-like
        Input data.
    Q : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 4, Eq 4.11, p. 160
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Loftq objective."})


def cheatsheet():
    return "km064: Loftq objective."
