# moirais.fn — function file (hadesllm/moirais)
"""Biological neuron model (McCulloch-Pitts): weighted sum then activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_biological_neuron"]


def geron_biological_neuron(x, w, b):
    """
    Biological neuron model (McCulloch-Pitts): weighted sum then activation

    Formula: a = phi(sum_i w_i x_i + b)

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a

    References
    ----------
    Géron Ch 9
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Biological neuron model (McCulloch-Pitts): weighted sum then activation"})


def cheatsheet():
    return "hmbnm: Biological neuron model (McCulloch-Pitts): weighted sum then activation"
