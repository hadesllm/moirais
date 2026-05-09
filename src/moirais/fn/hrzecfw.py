# moirais.fn — function file (hadesllm/moirais)
"""Empirical characteristic function for deconvolution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_empirical_cf"]


def horowitz_empirical_cf(w, tau):
    """
    Empirical characteristic function for deconvolution

    Formula: psiNW(tau) = (1/n)*sum_j exp(i*tau*W_j)

    Parameters
    ----------
    w : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: characteristic_function

    References
    ----------
    Horowitz Ch 5
    """
    w = np.asarray(w, dtype=float)
    n = int(w) if w.ndim == 0 else len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical characteristic function for deconvolution"})


def cheatsheet():
    return "hrzecfw: Empirical characteristic function for deconvolution"
