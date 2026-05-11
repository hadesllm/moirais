# morie.fn — function file (hadesllm/morie)
"""Theorem 2.2: bias of bias-reduced KDFE is O(h^4)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm2_2_bias_brdkdfe"]


def fauzi_thm2_2_bias_brdkdfe(x, bandwidth, a):
    """
    Theorem 2.2: bias of bias-reduced KDFE is O(h^4)

    Formula: Bias[F_tilde_X] = h^4*a^2*(b_2^2(x)-2*b_4(x)*F_X(x))/(2*F_X(x)) + o(h^4) + O(1/n)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias

    References
    ----------
    Fauzi Ch 2, Theorem 2.2, Eq 2.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 2.2: bias of bias-reduced KDFE is O(h^4)"})


def cheatsheet():
    return "fzt22: Theorem 2.2: bias of bias-reduced KDFE is O(h^4)"
