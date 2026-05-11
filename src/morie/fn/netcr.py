# morie.fn — function file (hadesllm/morie)
"""Partial correlation network (regularized via pseudo-inverse)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def network_correlation(
    data: pd.DataFrame | np.ndarray,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Patience is bitter, but its fruit is sweet. — Aristotle"""
    X = np.asarray(data, dtype=np.float64)
    R = np.corrcoef(X, rowvar=False)
    p = R.shape[0]

    try:
        P = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(R)

    # Convert precision to partial correlations
    D = np.sqrt(np.diag(P))
    D[D == 0] = 1.0
    pcor = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j:
                pcor[i, j] = -P[i, j] / (D[i] * D[j])

    if threshold > 0:
        pcor[np.abs(pcor) < threshold] = 0.0

    return pcor


def cheatsheet() -> str:
    return "network_correlation({}) -> Partial correlation network (regularized via pseudo-inverse)"
