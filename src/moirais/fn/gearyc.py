# moirais.fn — function file (hadesllm/moirais)
"""Geary's C with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def gearyc(x: Union[Sequence, np.ndarray],
           W: Union[Sequence, np.ndarray]):
    """Geary's C: alternative spatial autocorrelation."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    n = a.size
    if W.shape != (n, n):
        raise ValueError(f"W must be {n}x{n}, got {W.shape}.")
    S0 = W.sum()
    if S0 == 0:
        raise ValueError("W sums to zero.")
    diff_sq = (a[:, None] - a[None, :]) ** 2
    num = float((W * diff_sq).sum())
    denom = float(((a - a.mean()) ** 2).sum())
    if denom == 0:
        raise ValueError("zero variance.")
    C = ((n - 1) / (2 * S0)) * num / denom
    if C < 0.5: pattern = "strong positive autocorrelation (clustering)"
    elif C < 0.9: pattern = "weak positive autocorrelation"
    elif C <= 1.1: pattern = "no spatial autocorrelation (~random)"
    elif C <= 1.5: pattern = "weak negative autocorrelation"
    else: pattern = "strong negative autocorrelation (dispersion)"
    return RichResult(
        title="Geary's C spatial autocorrelation",
        summary_lines=[
            ("Geary's C", C),
            ("Expected C (null)", 1.0),
            ("Pattern", pattern),
            ("n locations", n),
        ],
        interpretation=(f"C = {C:.4f}; null = 1, range ~[0, 2]. {pattern.capitalize()}. "
                        "More sensitive to local outliers than Moran's I."),
        payload={"value": C, "statistic": C, "pattern": pattern},
    )
