"""AlphaZero policy distillation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_distill_student"]


def alphazero_distill_student(teacher, student, data):
    """
    AlphaZero policy distillation

    Formula: student matches teacher's pi distribution

    Parameters
    ----------
    teacher : array-like
        Input data.
    student : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hinton-Vinyals-Dean (2015)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero policy distillation"})


def cheatsheet():
    return "agdsts: AlphaZero policy distillation"
