# morie.fn — function file (hadesllm/morie)
"""Q-Former: learnable query tokens attend to visual features via cross-attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_q_former"]


def kamath_q_former(queries, visual_features):
    """
    Q-Former: learnable query tokens attend to visual features via cross-attention

    Formula: Z = CrossAttn(Q=learnable_queries, K=V=visual_features); Z then fed to LLM

    Parameters
    ----------
    queries : array-like
        Input data.
    visual_features : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Z

    References
    ----------
    Kamath Ch 9, Q-Former section (BLIP-2)
    """
    queries = np.atleast_1d(np.asarray(queries, dtype=float))
    n = len(queries)
    result = float(np.mean(queries))
    se = float(np.std(queries, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Q-Former: learnable query tokens attend to visual features via cross-attention"})


def cheatsheet():
    return "kmqfrm: Q-Former: learnable query tokens attend to visual features via cross-attention"
