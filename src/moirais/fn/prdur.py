# moirais.fn — function file (hadesllm/moirais)
"""PR interval/duration measurement."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "These aren't the droids you're looking for."


def pr_duration(p_on, qrs_on, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Measure PR interval from P-wave onset to QRS onset.

    Parameters
    ----------
    p_on : array-like of int
        P-wave onset sample indices.
    qrs_on : array-like of int
        QRS onset sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    p_on = np.asarray(p_on, dtype=int)
    qrs_on = np.asarray(qrs_on, dtype=int)
    n = min(len(p_on), len(qrs_on))
    if n == 0:
        return DescriptiveResult(
            name="pr_duration",
            value=0.0,
            extra={"pr_intervals": np.array([])},
        )
    pr = (qrs_on[:n] - p_on[:n]) / fs
    return DescriptiveResult(
        name="pr_duration",
        value=float(np.mean(pr)),
        extra={
            "pr_intervals": pr,
            "mean_pr": float(np.mean(pr)),
            "std_pr": float(np.std(pr, ddof=1)) if n > 1 else 0.0,
            "n_beats": n,
            "fs": fs,
        },
    )


prdur = pr_duration


def cheatsheet() -> str:
    return "pr_duration({}) -> PR interval/duration measurement."
