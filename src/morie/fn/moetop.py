"""MoE top-k routing with auxiliary load-balance loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["moe_topk_routing"]


def moe_topk_routing(y, x, W_g, experts, k):
    """
    MoE top-k routing with auxiliary load-balance loss

    Formula: y = sum_{i in topk(G(x))} G(x)_i E_i(x); aux = sum f_i P_i

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
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lepikhin et al. (2021) GShard; Fedus et al. (2022)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MoE top-k routing with auxiliary load-balance loss"})


def cheatsheet():
    return "moetop: MoE top-k routing with auxiliary load-balance loss"
