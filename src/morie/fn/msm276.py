"""Numbered display equation (14.8) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_8"]


def mvsml_convolutional_nn_eq_14_8(where, bc, j, T, jx, a):
    """Confine yourself to the present. -- Marcus Aurelius"""
    where = np.atleast_1d(np.asarray(where, dtype=float))
    n = len(where)
    result = float(np.mean(where))
    se = float(np.std(where, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.8) from MVSML chapter 14."})


def cheatsheet():
    return "msm276: Numbered display equation (14.8) from MVSML chapter 14."
