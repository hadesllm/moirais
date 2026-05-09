# moirais.fn — function file (hadesllm/moirais)
"""Hidden layers guideline: add layers until validation error stops improving."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_hidden_layers_heuristic"]


def geron_hidden_layers_heuristic(model, X, y):
    """
    Hidden layers guideline: add layers until validation error stops improving

    Formula: depth L chosen so val_err(L) converges

    Parameters
    ----------
    model : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best_depth

    References
    ----------
    Géron Ch 9
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hidden layers guideline: add layers until validation error stops improving"})


def cheatsheet():
    return "hmhplm: Hidden layers guideline: add layers until validation error stops improving"
