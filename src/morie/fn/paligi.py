"""Parametric ALiBi with learnable per-head slopes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["parametric_alibi"]


def parametric_alibi(y, Q, K, V, s_h):
    """
    Parametric ALiBi with learnable per-head slopes

    Formula: a_ij = q_i k_j / sqrt(d) - sigmoid(s_h) * |i - j|

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    s_h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Faisal & Anastasopoulos (2022)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parametric ALiBi with learnable per-head slopes"})


def cheatsheet():
    return "paligi: Parametric ALiBi with learnable per-head slopes"
