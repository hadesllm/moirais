"""Patience is bitter, but its fruit is sweet. — Aristotle"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def byzantine_detect(
    reports: np.ndarray,
    *,
    threshold: float = 2.0,
    method: str = "median",
) -> DescriptiveResult:
    """Detect Byzantine (faulty/adversarial) reporters in distributed systems.

    Each row is a reporter, each column is a measurement round.
    Byzantine reporters deviate from the consensus by more than ``threshold``
    MADs (median absolute deviations).

    Parameters
    ----------
    reports : ndarray of shape (n_reporters, n_rounds)
        Matrix of reported values.
    threshold : float
        Number of MADs beyond which a reporter is flagged.
    method : str
        Consensus method: 'median' or 'trimmed_mean'.

    Returns
    -------
    DescriptiveResult
        With ``value`` = boolean mask of detected Byzantines and
        ``extra`` containing deviation scores.
    """
    R = np.asarray(reports, dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    n_rep, n_rounds = R.shape

    if method == "median":
        consensus = np.median(R, axis=0)
    elif method == "trimmed_mean":
        sorted_r = np.sort(R, axis=0)
        trim = max(1, n_rep // 4)
        if n_rep - 2 * trim > 0:
            consensus = sorted_r[trim:-trim].mean(axis=0)
        else:
            consensus = sorted_r.mean(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    deviations = np.abs(R - consensus)
    mad = np.median(np.abs(R - np.median(R, axis=0)), axis=0)
    mad = np.where(mad < 1e-10, 1.0, mad)

    scores = np.mean(deviations / mad, axis=1)
    is_byzantine = scores > threshold

    return DescriptiveResult(
        name="byzantine_detect",
        value=is_byzantine,
        extra={
            "scores": scores,
            "n_detected": int(is_byzantine.sum()),
            "threshold": threshold,
            "n_reporters": n_rep,
            "n_rounds": n_rounds,
            "method": method,
        },
    )


stscm = byzantine_detect


def cheatsheet() -> str:
    return "byzantine_detect({}) -> Byzantine fault detection. 'I am the true leader!' -- Starsc"
