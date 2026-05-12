# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Adjusted Rand index."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def adjusted_rand_index(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> DescriptiveResult:
    """Adjusted Rand Index (ARI) between two clusterings.

    Parameters
    ----------
    labels_true : ndarray (n,)
        Ground truth labels.
    labels_pred : ndarray (n,)
        Predicted labels.

    Returns
    -------
    DescriptiveResult
        ``value`` is the ARI in [-1, 1].
    """
    y = np.asarray(labels_true)
    yp = np.asarray(labels_pred)
    n = len(y)

    classes_t = np.unique(y)
    classes_p = np.unique(yp)

    contingency = np.zeros((len(classes_t), len(classes_p)), dtype=np.int64)
    for i, ct in enumerate(classes_t):
        for j, cp in enumerate(classes_p):
            contingency[i, j] = int(np.sum((y == ct) & (yp == cp)))

    def _comb2(x):
        return x * (x - 1) / 2

    sum_comb_nij = np.sum(_comb2(contingency))
    sum_comb_ai = np.sum(_comb2(contingency.sum(axis=1)))
    sum_comb_bj = np.sum(_comb2(contingency.sum(axis=0)))
    comb_n = _comb2(n)

    expected = sum_comb_ai * sum_comb_bj / comb_n if comb_n > 0 else 0
    max_index = (sum_comb_ai + sum_comb_bj) / 2
    denom = max_index - expected

    if denom == 0:
        ari = 0.0 if sum_comb_nij == expected else 1.0
    else:
        ari = (sum_comb_nij - expected) / denom

    return DescriptiveResult(
        name="AdjustedRandIndex",
        value=float(ari),
    )


arisc = adjusted_rand_index


def cheatsheet() -> str:
    return "adjusted_rand_index({}) -> Adjusted Rand index."
