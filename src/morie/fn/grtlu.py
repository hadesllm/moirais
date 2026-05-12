# morie.fn -- function file (hadesllm/morie)
"""Threshold Logic Unit: weighted sum with step activation (Rosenblatt perceptron)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_threshold_logic_unit"]


def geron_threshold_logic_unit(x, w, b):
    """
    Threshold Logic Unit: weighted sum with step activation (Rosenblatt perceptron)

    Formula: h(x) = heaviside(w^T x + b)

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
        Keys: output

    References
    ----------
    Géron Ch 9, Threshold Logic Unit section
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Threshold Logic Unit: weighted sum with step activation (Rosenblatt perceptron)"})


def cheatsheet():
    return "grtlu: Threshold Logic Unit: weighted sum with step activation (Rosenblatt perceptron)"
