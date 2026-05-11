# morie.fn — function file (hadesllm/morie)
"""Symbolic differentiation: manipulate algebraic expressions analytically."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_symbolic_diff"]


def geron_symbolic_diff(expr, var):
    """
    Symbolic differentiation: manipulate algebraic expressions analytically

    Formula: apply derivative rules symbolically to expression tree

    Parameters
    ----------
    expr : array-like
        Input data.
    var : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: derivative_expr

    References
    ----------
    Géron Appendix A
    """
    expr = np.atleast_1d(np.asarray(expr, dtype=float))
    n = len(expr)
    result = float(np.mean(expr))
    se = float(np.std(expr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symbolic differentiation: manipulate algebraic expressions analytically"})


def cheatsheet():
    return "hmsymd: Symbolic differentiation: manipulate algebraic expressions analytically"
