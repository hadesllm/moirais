# moirais.fn — function file (hadesllm/moirais)
"""Mutual information with R-style verbose result."""

from typing import Sequence, Union
import math
import numpy as np


def muthi(joint: Union[Sequence, np.ndarray]):
    """MI from joint p(x,y): I(X;Y) = Sigma_ij P_ij log(P_ij / (P_i. P.j))."""
    from ._richresult import RichResult
    P = np.asarray(joint, dtype=float)
    P = P / P.sum()
    Px = P.sum(axis=1); Py = P.sum(axis=0)
    out = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0 and Px[i] > 0 and Py[j] > 0:
                out += P[i, j] * math.log(P[i, j] / (Px[i] * Py[j]))
    Hx = -np.sum([p * math.log(p) for p in Px if p > 0])
    Hy = -np.sum([p * math.log(p) for p in Py if p > 0])
    norm = out / min(Hx, Hy) if min(Hx, Hy) > 0 else float("nan")
    return RichResult(
        title="Mutual information (discrete, MLE)",
        summary_lines=[
            ("I(X;Y)", float(out)),
            ("Normalized I / min(H(X), H(Y))", float(norm)),
            ("H(X)", float(Hx)),
            ("H(Y)", float(Hy)),
            ("Joint shape", P.shape),
        ],
        interpretation=(f"I={out:.4f} nats. Normalized = {norm:+.3f} on [0,1] scale; "
                        "0 = independent, 1 = perfect mutual determination."),
        payload={"value": float(out), "statistic": float(out),
                 "normalized": float(norm), "H_x": float(Hx), "H_y": float(Hy)},
    )
