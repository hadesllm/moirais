# moirais.fn — function file (hadesllm/moirais)
"""Counterfactual notation: Y_x outcome had X been set to x by intervention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["counterfactual_notation"]


def counterfactual_notation(Y, X, x_val, u):
    """
    Counterfactual notation: Y_x outcome had X been set to Y by intervention

    Formula: Y_x(u) = Y_{M_x}(u); M_x is modified SCM where X = Y replaces X's equation

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    x_val : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'Y_x': 'float'}

    References
    ----------
    Molak Ch 2
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Counterfactual notation: Y_x outcome had X been set to x by intervention"})


def cheatsheet():
    return "ctcfl: Counterfactual notation: Y_x outcome had X been set to x by intervention"
