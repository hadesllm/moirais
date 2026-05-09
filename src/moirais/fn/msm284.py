"""Numbered display equation (14.12) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_12"]


def mvsml_convolutional_nn_eq_14_12(SSE, j, where, X, T, P):
    """You have power over your mind — not outside events. — Marcus Aurelius"""
    SSE = np.atleast_1d(np.asarray(SSE, dtype=float))
    n = len(SSE)
    result = float(np.mean(SSE))
    se = float(np.std(SSE, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.12) from MVSML chapter 14."})


def cheatsheet():
    return "msm284: Numbered display equation (14.12) from MVSML chapter 14."
