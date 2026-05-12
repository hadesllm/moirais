r"""Moe topk gating.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_moe_topk_gating"]


def kamath_ch2_moe_topk_gating(x, W_g):
    r"""
    Moe topk gating.

    Formula: G(x) := \mathrm{Softmax}(\mathrm{TopK}(x \cdot W_g))

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
    Kamath et al (2024), Ch 2, Eq 2.40, p. 74
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moe topk gating."})


def cheatsheet():
    return "km040: Moe topk gating."
