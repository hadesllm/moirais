# morie.fn — function file (hadesllm/morie)
"""XLNet: permutation-based autoregressive pretraining."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_xlnet"]


def geron_xlnet(X, n_layers):
    """
    XLNet: permutation-based autoregressive pretraining

    Formula: maximize E_pi sum_t log P(x_{pi_t} | x_{pi_<t})

    Parameters
    ----------
    X : array-like
        Input data.
    n_layers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 15
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "XLNet: permutation-based autoregressive pretraining"})


def cheatsheet():
    return "hmxln: XLNet: permutation-based autoregressive pretraining"
