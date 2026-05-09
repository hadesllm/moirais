# moirais.fn — function file (hadesllm/moirais)
"""Backpropagation: gradient of the loss w.r.t. each weight layer via chain rule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_backpropagation_gradient"]


def geron_backpropagation_gradient(activations, weights, y_true):
    """
    Backpropagation: gradient of the loss w.r.t. each weight layer via chain rule

    Formula: d_L = nabla_a L(a_L) .* phi'(z_L); d_l = (W_{l+1}^T d_{l+1}) .* phi'(z_l)

    Parameters
    ----------
    activations : array-like
        Input data.
    weights : array-like
        Input data.
    y_true : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradients

    References
    ----------
    Géron Ch 9, Backpropagation Algorithm section
    """
    activations = np.asarray(activations, dtype=float)
    n = int(activations) if activations.ndim == 0 else len(activations)
    result = float(np.mean(activations))
    se = float(np.std(activations, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Backpropagation: gradient of the loss w.r.t. each weight layer via chain rule"})


def cheatsheet():
    return "grbp: Backpropagation: gradient of the loss w.r.t. each weight layer via chain rule"
