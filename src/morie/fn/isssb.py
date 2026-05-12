# morie.fn -- function file (hadesllm/morie)
"""Extract a subset of issue scale columns."""

from __future__ import annotations

from ._containers import DescriptiveResult


def issue_subset_extract(data, columns) -> DescriptiveResult:
    """Extract a subset of issue-scale columns from a data matrix.

    :param data: Full data matrix or dict-like.
    :param columns: List of column indices or names.
    :return: DescriptiveResult with extracted subset.

    .. epigraph:: "The Rumbling." -- Eren Yeager, Attack on Titan
    """
    import numpy as np

    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    cols = list(columns)
    subset = X[:, cols]
    return DescriptiveResult(
        name="issue_subset_extract",
        value=len(cols),
        extra={"subset": subset.tolist(), "columns": cols, "n_rows": subset.shape[0]},
    )


isssb = issue_subset_extract


def cheatsheet() -> str:
    return "issue_subset_extract({}) -> Extract a subset of issue scale columns."
