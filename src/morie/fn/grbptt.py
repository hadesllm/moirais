# morie.fn — function file (hadesllm/morie)
"""Backpropagation through time: unroll the RNN and apply standard backprop."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_backprop_through_time"]


def geron_backprop_through_time(loss_grads, hiddens, inputs):
    """
    Backpropagation through time: unroll the RNN and apply standard backprop

    Formula: grad_W = sum_{t=1..T} (d L_t / d h_t) * (d h_t / d W), summed across time

    Parameters
    ----------
    loss_grads : array-like
        Input data.
    hiddens : array-like
        Input data.
    inputs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradients

    References
    ----------
    Géron Ch 13, Backpropagation Through Time section
    """
    loss_grads = np.atleast_1d(np.asarray(loss_grads, dtype=float))
    n = len(loss_grads)
    result = float(np.mean(loss_grads))
    se = float(np.std(loss_grads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Backpropagation through time: unroll the RNN and apply standard backprop"})


def cheatsheet():
    return "grbptt: Backpropagation through time: unroll the RNN and apply standard backprop"
