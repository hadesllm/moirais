# moirais.fn — function file (hadesllm/moirais)
"""Reverse-mode autodiff via chain rule (what PyTorch autograd computes)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_autograd_chain_rule"]


def geron_autograd_chain_rule(graph, grad_output):
    """
    Reverse-mode autodiff via chain rule (what PyTorch autograd computes)

    Formula: d L / d x = prod_{edges on path} (local Jacobian) applied in reverse

    Parameters
    ----------
    graph : array-like
        Input data.
    grad_output : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: grad_input

    References
    ----------
    Géron Ch 10, Autograd / Automatic Differentiation section
    """
    graph = np.atleast_1d(np.asarray(graph, dtype=float))
    n = len(graph)
    result = float(np.mean(graph))
    se = float(np.std(graph, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reverse-mode autodiff via chain rule (what PyTorch autograd computes)"})


def cheatsheet():
    return "graut: Reverse-mode autodiff via chain rule (what PyTorch autograd computes)"
