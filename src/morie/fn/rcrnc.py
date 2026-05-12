# morie.fn -- function file (hadesllm/morie)
"""Recurrence quantification analysis (RQA).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 15.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['rcrnc']

_QUOTE = "History repeats itself. -- C-3PO"


def rcrnc(
    x: np.ndarray,
    *,
    m: int = 3,
    tau: int = 1,
    epsilon: float | None = None,
) -> DescriptiveResult:
    """Recurrence quantification analysis.

    Constructs the recurrence matrix and computes RR (recurrence rate),
    DET (determinism), and L_max (longest diagonal).

    Parameters
    ----------
    x : array-like
        1-D time series.
    m : int
        Embedding dimension.
    tau : int
        Time delay.
    epsilon : float or None
        Recurrence threshold (default 10% of max distance).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    N = n - (m - 1) * tau
    if N < 5:
        raise ValueError("Signal too short.")

    Y = np.array([x[i:i + (m - 1) * tau + 1:tau] for i in range(N)])

    dist = np.zeros((N, N))
    for i in range(N):
        dist[i] = np.max(np.abs(Y - Y[i]), axis=1)

    if epsilon is None:
        epsilon = 0.1 * np.max(dist)

    R = (dist <= epsilon).astype(int)
    np.fill_diagonal(R, 0)

    rr = float(np.sum(R)) / (N * (N - 1)) if N > 1 else 0.0

    diag_lengths = []
    for k in range(1, N):
        d = np.diag(R, k)
        length = 0
        for v in d:
            if v:
                length += 1
            else:
                if length >= 2:
                    diag_lengths.append(length)
                length = 0
        if length >= 2:
            diag_lengths.append(length)

    if diag_lengths:
        det_sum = sum(diag_lengths)
        total_rec = np.sum(np.triu(R, k=1))
        det = det_sum / total_rec if total_rec > 0 else 0.0
        l_max = max(diag_lengths)
        l_mean = float(np.mean(diag_lengths))
    else:
        det = 0.0
        l_max = 0
        l_mean = 0.0

    return DescriptiveResult(
        name="rcrnc",
        value=rr,
        extra={
            "recurrence_rate": rr,
            "determinism": float(det),
            "l_max": l_max,
            "l_mean": l_mean,
            "recurrence_matrix": R,
            "epsilon": epsilon,
        },
    )


def cheatsheet() -> str:
    return "rcrnc({}) -> Recurrence quantification analysis."
