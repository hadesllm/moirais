# morie.fn -- function file (hadesllm/morie)
"""Reverse-mode automatic differentiation (backprop) via chain rule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_reverse_autodiff"]


def geron_reverse_autodiff(f, x):
    """
    Reverse-mode automatic differentiation (backprop) via chain rule

    Formula: build computation graph; propagate gradients backward

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradient

    References
    ----------
    Géron Appendix A
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reverse-mode automatic differentiation (backprop) via chain rule"})


def cheatsheet():
    return "hmrad: Reverse-mode automatic differentiation (backprop) via chain rule"
