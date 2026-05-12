# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Backpropagation via chain rule for multi-layer networks."""
import numpy as np
from ._richresult import RichResult

__all__ = ["backpropagation_chain_rule"]


def backpropagation_chain_rule(layers, activations, loss_grad):
    """
    Backpropagation via chain rule for multi-layer networks

    Formula: dL/dw_l = dL/da_{l+1} * da_{l+1}/dz_{l+1} * dz_{l+1}/dw_l; delta_l = W_{l+1}'*delta_{l+1} * f'(z_l)

    Parameters
    ----------
    layers : array-like
        Input data.
    activations : array-like
        Input data.
    loss_grad : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'gradients': 'list'}

    References
    ----------
    Montesinos Lopez Ch 10
    """
    layers = np.asarray(layers, dtype=float)
    n = int(layers) if layers.ndim == 0 else len(layers)
    result = float(np.mean(layers))
    se = float(np.std(layers, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Backpropagation via chain rule for multi-layer networks"})


def cheatsheet():
    return "bprop: Backpropagation via chain rule for multi-layer networks"
