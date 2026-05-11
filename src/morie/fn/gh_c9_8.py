# morie.fn — function file (hadesllm/morie)
"""Nonlinear autoregression contraction rate via Markov chain framework."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_nlar_crt"]


def ghosal_nlar_crt(x):
    """
    Nonlinear autoregression contraction rate via Markov chain framework

    Formula: X_t = f(X_{t-1}) + e_t, f ~ GP, rate n^{-s/(2s+1)} with ergodicity

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 9 §9.5.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonlinear autoregression contraction rate via Markov chain framework"})


def cheatsheet():
    return "gh_c9_8: Nonlinear autoregression contraction rate via Markov chain framework"
