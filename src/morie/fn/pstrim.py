# morie.fn -- function file (hadesllm/morie)
"""Propensity score trimming."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def propensity_trim(
    ps_scores: np.ndarray,
    *,
    trim: float = 0.01,
) -> DescriptiveResult:
    """Trim extreme propensity scores for positivity.

    Parameters
    ----------
    ps_scores : (n,) array
    trim : float
        Symmetric trim fraction (e.g. 0.01 trims below 1st and above 99th pctile).

    Returns
    -------
    DescriptiveResult
    """
    ps = np.asarray(ps_scores, dtype=float).ravel()
    n = len(ps)

    lo = np.percentile(ps, 100 * trim)
    hi = np.percentile(ps, 100 * (1 - trim))
    mask = (ps >= lo) & (ps <= hi)
    n_kept = int(mask.sum())
    n_trimmed = n - n_kept

    return DescriptiveResult(
        name="ps_trim",
        value=float(n_kept),
        extra={
            "n_original": n,
            "n_trimmed": n_trimmed,
            "lo": float(lo),
            "hi": float(hi),
            "trim": trim,
            "kept_indices": np.where(mask)[0].tolist(),
        },
    )


pstrim = propensity_trim


def cheatsheet() -> str:
    return "propensity_trim({}) -> Propensity score trimming."
