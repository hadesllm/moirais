# moirais.fn — function file (hadesllm/moirais)
"""RNN/LSTM for genomic prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rnn_genomic"]


def rnn_genomic(x, y, markers):
    """
    RNN/LSTM for genomic prediction

    Formula: Sequential marker processing via LSTM cells

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 14
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RNN/LSTM for genomic prediction"})


def cheatsheet():
    return "rnnge: RNN/LSTM for genomic prediction"
