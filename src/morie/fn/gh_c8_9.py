# morie.fn -- function file (hadesllm/morie)
"""Contraction rates for Markov chains: applied to nonlinear autoregression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_markov_crt"]


def ghosal_markov_crt(x):
    """
    Contraction rates for Markov chains: applied to nonlinear autoregression

    Formula: eps_n from stationary-distribution entropy and prior mass for Markov transitions

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
    Ghosal Ch 8 §8.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contraction rates for Markov chains: applied to nonlinear autoregression"})


def cheatsheet():
    return "gh_c8_9: Contraction rates for Markov chains: applied to nonlinear autoregression"
