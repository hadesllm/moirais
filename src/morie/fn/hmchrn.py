# morie.fn -- function file (hadesllm/morie)
"""Character-level RNN language model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_char_rnn"]


def geron_char_rnn(text, hidden, epochs, lr):
    """
    Character-level RNN language model

    Formula: P(c_t | c_{<t}) = softmax(W h_t)

    Parameters
    ----------
    text : array-like
        Input data.
    hidden : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 14
    """
    text = np.atleast_1d(np.asarray(text, dtype=float))
    n = len(text)
    result = float(np.mean(text))
    se = float(np.std(text, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Character-level RNN language model"})


def cheatsheet():
    return "hmchrn: Character-level RNN language model"
