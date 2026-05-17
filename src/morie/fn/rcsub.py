# morie.fn -- function file (hadesllm/morie)
"""Subset roll calls by margin threshold."""

from __future__ import annotations

from ._containers import DescriptiveResult


def subset_roll_calls(votes, min_margin=0.025) -> DescriptiveResult:
    """Filter out lopsided votes where margin < min_margin.

    .. epigraph:: We must know. We will know. -- David Hilbert
    """
    import numpy as np

    votes = np.asarray(votes, dtype=float)
    if votes.ndim == 1:
        votes = votes.reshape(1, -1)
    n_leg = votes.shape[0]
    margins = []
    keep = []
    for j in range(votes.shape[1]):
        col = votes[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        yea_frac = np.mean(valid)
        margin = abs(yea_frac - 0.5)
        margins.append(margin)
        if margin >= min_margin:
            keep.append(j)
    filtered = votes[:, keep] if keep else np.empty((n_leg, 0))
    return DescriptiveResult(
        name="subset_roll_calls",
        value=float(len(keep)),
        extra={
            "n_kept": len(keep),
            "n_dropped": votes.shape[1] - len(keep),
            "kept_indices": keep,
            "filtered_matrix": filtered,
        },
    )


rcsub = subset_roll_calls


def cheatsheet() -> str:
    return "subset_roll_calls({}) -> Subset roll calls by margin threshold."
