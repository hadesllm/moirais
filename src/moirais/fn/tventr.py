# moirais.fn — function file (hadesllm/moirais)
"""Total variation distance with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def tventr(p: Union[Sequence, np.ndarray],
           q: Union[Sequence, np.ndarray]):
    """Total variation distance: 0.5 * Sigma |p_i - q_i|."""
    from ._richresult import RichResult
    P = np.asarray(p, dtype=float); Q = np.asarray(q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: {P.shape} vs {Q.shape}.")
    P = P / P.sum(); Q = Q / Q.sum()
    tv = float(0.5 * np.sum(np.abs(P - Q)))
    return RichResult(
        title="Total variation distance",
        summary_lines=[
            ("TV(P, Q)", tv),
            ("Range", "[0, 1]"),
            ("Support size", int(P.size)),
        ],
        interpretation=(f"TV={tv:.4f}; 0 = identical distributions, 1 = disjoint "
                        "supports. Symmetric (TV(P,Q) = TV(Q,P)). For asymmetric "
                        "comparison use `kldivg`."),
        payload={"value": tv, "statistic": tv},
    )
