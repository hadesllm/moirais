# moirais.fn — function file (hadesllm/moirais)
"""Per-expert token capacity with capacity factor C."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_expert_capacity_factor"]


def kamath_expert_capacity_factor(tokens_per_batch, num_experts, C):
    """
    Per-expert token capacity with capacity factor C

    Formula: capacity = C * (tokens_per_batch / num_experts); drop overflow

    Parameters
    ----------
    tokens_per_batch : array-like
        Input data.
    num_experts : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: capacity

    References
    ----------
    Kamath Ch 2, Capacity Factor section
    """
    tokens_per_batch = np.atleast_1d(np.asarray(tokens_per_batch, dtype=float))
    n = len(tokens_per_batch)
    result = float(np.mean(tokens_per_batch))
    se = float(np.std(tokens_per_batch, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Per-expert token capacity with capacity factor C"})


def cheatsheet():
    return "kmcap: Per-expert token capacity with capacity factor C"
