# morie.fn — function file (hadesllm/morie)
"""Kullback-Leibler divergence with R-style verbose result."""

from typing import Sequence, Union
import math
import numpy as np


def kldivg(p: Union[Sequence[float], np.ndarray],
           q: Union[Sequence[float], np.ndarray],
           base: float = 2.0):
    """KL(P||Q) = Sigma p_i log(p_i / q_i)."""
    from ._richresult import RichResult
    P = np.asarray(p, dtype=float); Q = np.asarray(q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: {P.shape} vs {Q.shape}.")
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError("probabilities must be non-negative.")
    P = P / P.sum(); Q = Q / Q.sum()
    mask = P > 0
    warnings = []
    if np.any((Q == 0) & mask):
        warnings.append("Q has zero where P>0 - KL is infinite (no shared support).")
        v = float("inf")
    else:
        v = float(np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask]))) / math.log(base))
    return RichResult(
        title="Kullback-Leibler divergence (discrete)",
        summary_lines=[
            (f"KL(P || Q), base {base}", v),
            ("Support size", int(P.size)),
            ("P entropy", float(-np.sum(P[mask] * np.log(P[mask])) / math.log(base))),
            ("Q entropy", float(-np.sum(Q[Q > 0] * np.log(Q[Q > 0])) / math.log(base))),
        ],
        warnings=warnings,
        interpretation=("KL is asymmetric (KL(P||Q) != KL(Q||P)). 0 means P=Q. "
                        "Higher = more divergent. Use total variation (`tventr`) for "
                        "symmetric distance."),
        payload={"value": v, "statistic": v},
    )
