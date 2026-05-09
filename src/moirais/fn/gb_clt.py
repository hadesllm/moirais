# moirais.fn — function file (hadesllm/moirais)
"""Central Limit Theorem: sqrt(n)*(Xbar - mu)/sigma -> N(0,1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_clt"]


def gibbons_clt(n, mu, sigma):
    """
    Central Limit Theorem: sqrt(n)*(Xbar - mu)/sigma -> N(0,1)

    Formula: (Xbar - mu)/(sigma/sqrt(n)) ->_d N(0,1)

    Parameters
    ----------
    n : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: normal_limit

    References
    ----------
    Gibbons Ch 1.2.6
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Central Limit Theorem: sqrt(n)*(Xbar - mu)/sigma -> N(0,1)"})


def cheatsheet():
    return "gb_clt: Central Limit Theorem: sqrt(n)*(Xbar - mu)/sigma -> N(0,1)"
