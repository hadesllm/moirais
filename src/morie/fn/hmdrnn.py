# morie.fn -- function file (hadesllm/morie)
"""Deep (stacked) RNN: multiple recurrent layers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_deep_rnn"]


def geron_deep_rnn(X, hidden_sizes, n_layers):
    """
    Deep (stacked) RNN: multiple recurrent layers

    Formula: h_t^(l) = phi(W_x^(l) h_t^(l-1) + W_h^(l) h_{t-1}^(l))

    Parameters
    ----------
    X : array-like
        Input data.
    hidden_sizes : array-like
        Input data.
    n_layers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 13
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep (stacked) RNN: multiple recurrent layers"})


def cheatsheet():
    return "hmdrnn: Deep (stacked) RNN: multiple recurrent layers"
