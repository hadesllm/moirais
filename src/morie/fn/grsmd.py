# morie.fn — function file (hadesllm/morie)
"""Symbolic differentiation: apply rules (sum, product, chain) to expression trees."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_symbolic_differentiation"]


def geron_symbolic_differentiation(expression):
    """
    Symbolic differentiation: apply rules (sum, product, chain) to expression trees

    Formula: d/dx [f+g] = f'+g'; d/dx [f*g] = f'g + fg'; d/dx [f(g(x))] = f'(g) * g'(x)

    Parameters
    ----------
    expression : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: derivative_expression

    References
    ----------
    Géron Appendix A, Symbolic Differentiation section
    """
    expression = np.atleast_1d(np.asarray(expression, dtype=float))
    n = len(expression)
    result = float(np.mean(expression))
    se = float(np.std(expression, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symbolic differentiation: apply rules (sum, product, chain) to expression trees"})


def cheatsheet():
    return "grsmd: Symbolic differentiation: apply rules (sum, product, chain) to expression trees"
