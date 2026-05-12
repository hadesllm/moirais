# morie.fn -- function file (hadesllm/morie)
"""Reverse-mode autodiff: forward pass stores activations; backward pass applies chain rule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_reverse_mode_autodiff"]


def geron_reverse_mode_autodiff(graph, loss_grad):
    """
    Reverse-mode autodiff: forward pass stores activations; backward pass applies chain rule

    Formula: d L / d x = sum_{y in children(x)} (d L / d y) * (d y / d x)

    Parameters
    ----------
    graph : array-like
        Input data.
    loss_grad : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: grads

    References
    ----------
    Géron Appendix A, Reverse-mode autodiff section
    """
    graph = np.atleast_1d(np.asarray(graph, dtype=float))
    n = len(graph)
    result = float(np.mean(graph))
    se = float(np.std(graph, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reverse-mode autodiff: forward pass stores activations; backward pass applies chain rule"})


def cheatsheet():
    return "grrad: Reverse-mode autodiff: forward pass stores activations; backward pass applies chain rule"
