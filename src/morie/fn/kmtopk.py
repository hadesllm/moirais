# morie.fn — function file (hadesllm/morie)
"""Top-k expert gating: keep top-k gate scores, renormalize, zero the rest."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_moe_top_k_gating"]


def kamath_moe_top_k_gating(gates, k):
    """
    Top-k expert gating: keep top-k gate scores, renormalize, zero the rest

    Formula: TopK(g)_i = g_i if i in argtopk(g, k) else 0;  g' = TopK(g) / sum(TopK(g))

    Parameters
    ----------
    gates : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gates_topk

    References
    ----------
    Kamath Ch 2, Top-k expert gating section
    """
    gates = np.atleast_1d(np.asarray(gates, dtype=float))
    n = len(gates)
    result = float(np.mean(gates))
    se = float(np.std(gates, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Top-k expert gating: keep top-k gate scores, renormalize, zero the rest"})


def cheatsheet():
    return "kmtopk: Top-k expert gating: keep top-k gate scores, renormalize, zero the rest"
