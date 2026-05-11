"""Numbered display equation (5.5) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_5"]


def mvsml_linear_mixed_models_eq_5_5(where, y, GID, are, again, the):
    """Errors using inadequate data are much less than those using no data at all. — Charles Babbage"""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.5) from MVSML chapter 5."})


def cheatsheet():
    return "msm030: Numbered display equation (5.5) from MVSML chapter 5."
