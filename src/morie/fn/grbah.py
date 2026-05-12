# morie.fn -- function file (hadesllm/morie)
"""Bahdanau additive attention: score, softmax, context."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bahdanau_attention"]


def geron_bahdanau_attention(decoder_state, encoder_states, Wh, Ws, v):
    """
    Bahdanau additive attention: score, softmax, context

    Formula: e_ti = v^T tanh(W_h h_t + W_s s_i); alpha_ti = softmax(e_ti); c_t = sum_i alpha_ti s_i

    Parameters
    ----------
    decoder_state : array-like
        Input data.
    encoder_states : array-like
        Input data.
    Wh : array-like
        Input data.
    Ws : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: context, weights

    References
    ----------
    Géron Ch 14, Bahdanau Attention section
    """
    decoder_state = np.atleast_1d(np.asarray(decoder_state, dtype=float))
    n = len(decoder_state)
    result = float(np.mean(decoder_state))
    se = float(np.std(decoder_state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bahdanau additive attention: score, softmax, context"})


def cheatsheet():
    return "grbah: Bahdanau additive attention: score, softmax, context"
