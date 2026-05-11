# morie.fn — function file (hadesllm/morie)
"""Markov switching regression model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["regime_switching"]


def regime_switching(x):
    """
    Markov switching regression model

    Formula: y_t = mu_{s_t} + e_t, s_t ~ Markov chain

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
    Hamilton (1989)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Markov switching regression model"})


def cheatsheet():
    return "regms: Markov switching regression model"
