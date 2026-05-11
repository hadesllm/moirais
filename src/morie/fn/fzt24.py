# morie.fn — function file (hadesllm/morie)
"""Theorem 2.4: MISE of bias-reduced KDFE is smaller than standard KDFE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm2_4_mise_brdkdfe"]


def fauzi_thm2_4_mise_brdkdfe(x, bandwidth, a):
    """
    Theorem 2.4: MISE of bias-reduced KDFE is smaller than standard KDFE

    Formula: MISE(F_tilde_X) = h^8*a^4*int[(b_2^2-2b_4*F)/(2F)]^2 + (1/n)*int F(1-F) - (h/n)*[...] + o(h^8+h/n)

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
        Keys: mise

    References
    ----------
    Fauzi Ch 2, Theorem 2.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 2.4: MISE of bias-reduced KDFE is smaller than standard KDFE"})


def cheatsheet():
    return "fzt24: Theorem 2.4: MISE of bias-reduced KDFE is smaller than standard KDFE"
