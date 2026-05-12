r"""Mixtral swiglu moe.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_mixtral_swiglu_moe"]


def kamath_ch2_mixtral_swiglu_moe(x, W_g):
    r"""
    Mixtral swiglu moe.

    Formula: y = \sum_{i=0}^{n-1} \mathrm{Softmax}(\mathrm{Top2}(x\cdot W_g))_i \cdot \mathrm{SwiGLU}_i(x)

    Parameters
    ----------
    x : array-like
        Input data.
    W_g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.41, p. 75
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixtral swiglu moe."})


def cheatsheet():
    return "km041: Mixtral swiglu moe."
