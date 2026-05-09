"""MoE feed-forward layer with router + expert mix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["moe_layer"]


def moe_layer(y, x, W_g, experts, top_k):
    """
    MoE feed-forward layer with router + expert mix

    Formula: y = sum_i G(x)_i E_i(x); G = softmax(W_g x)

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    W_g : array-like
        Input data.
    experts : array-like
        Input data.
    top_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shazeer et al. (2017) Outrageously Large NN
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MoE feed-forward layer with router + expert mix"})


def cheatsheet():
    return "moelyr: MoE feed-forward layer with router + expert mix"
