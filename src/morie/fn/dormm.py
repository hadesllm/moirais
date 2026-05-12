# morie.fn -- function file (hadesllm/morie)
"""Recurrence quantification analysis. 'Dormammu, I have come to bargain.' -- Doctor Strange"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def recurrence_quantification(
    x: np.ndarray | list[float],
    *,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold: float | None = None,
    threshold_pct: float = 10.0,
) -> DescriptiveResult:
    """Perform recurrence quantification analysis (RQA) on a time series.

    Constructs the recurrence matrix from time-delay embedding and computes
    RQA measures: recurrence rate (RR), determinism (DET), average diagonal
    line length (L), entropy (ENTR), laminarity (LAM).

    Parameters
    ----------
    x : array-like
        Univariate time series.
    embedding_dim : int
        Embedding dimension.
    delay : int
        Time delay for embedding.
    threshold : float or None
        Fixed recurrence threshold. If None, uses *threshold_pct* percentile.
    threshold_pct : float
        Percentile of distance distribution for threshold (if threshold is None).

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``RR``, ``DET``, ``L``, ``ENTR``, ``LAM``,
        ``recurrence_matrix``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    N = len(x)
    M = N - (embedding_dim - 1) * delay
    if M < 5:
        raise ValueError("Time series too short for given embedding parameters")

    embedded = np.array([x[i * delay : i * delay + M] for i in range(embedding_dim)]).T

    dists = np.sqrt(np.sum((embedded[:, None, :] - embedded[None, :, :]) ** 2, axis=2))

    if threshold is None:
        threshold = float(np.percentile(dists[np.triu_indices(M, k=1)], threshold_pct))

    R = (dists <= threshold).astype(int)
    np.fill_diagonal(R, 0)

    n_recur = R.sum()
    RR = float(n_recur) / (M * (M - 1)) if M > 1 else 0.0

    diag_lengths = []
    for k in range(1, M):
        diag = np.diag(R, k)
        length = 0
        for v in diag:
            if v == 1:
                length += 1
            elif length > 0:
                diag_lengths.append(length)
                length = 0
        if length > 0:
            diag_lengths.append(length)

    dl = np.array(diag_lengths) if diag_lengths else np.array([0])
    det_sum = int(dl[dl >= 2].sum()) if len(dl) > 0 else 0
    total_recur = int(dl.sum()) if len(dl) > 0 else 0
    DET = float(det_sum / total_recur) if total_recur > 0 else 0.0
    L_mean = float(dl[dl >= 2].mean()) if np.any(dl >= 2) else 0.0

    if np.any(dl >= 2):
        lengths, counts = np.unique(dl[dl >= 2], return_counts=True)
        p = counts / counts.sum()
        ENTR = float(-np.sum(p * np.log(p + 1e-30)))
    else:
        ENTR = 0.0

    vert_lengths = []
    for j in range(M):
        length = 0
        for i in range(M):
            if R[i, j] == 1:
                length += 1
            elif length > 0:
                vert_lengths.append(length)
                length = 0
        if length > 0:
            vert_lengths.append(length)

    vl = np.array(vert_lengths) if vert_lengths else np.array([0])
    lam_sum = int(vl[vl >= 2].sum()) if len(vl) > 0 else 0
    total_vert = int(vl.sum()) if len(vl) > 0 else 0
    LAM = float(lam_sum / total_vert) if total_vert > 0 else 0.0

    return DescriptiveResult(
        name="recurrence_quantification",
        value={
            "RR": RR,
            "DET": DET,
            "L": L_mean,
            "ENTR": ENTR,
            "LAM": LAM,
            "recurrence_matrix": R,
        },
        extra={"M": M, "threshold": threshold, "embedding_dim": embedding_dim},
    )


dormm = recurrence_quantification


def cheatsheet() -> str:
    return "recurrence_quantification({}) -> Recurrence quantification analysis. 'Dormammu, I have come t"
