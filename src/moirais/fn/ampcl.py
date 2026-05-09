# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Classify signal segments by amplitude level."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Stay on target."


def amplitude_classify(x, thresholds=None, **kwargs) -> DescriptiveResult:
    """Classify signal segments by amplitude level.

    Parameters
    ----------
    x : array-like
        Input signal.
    thresholds : array-like or None
        Boundaries between amplitude classes. If ``None``, terciles are used.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)

    if thresholds is None:
        thresholds = np.percentile(np.abs(x), [33, 67])
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    labels = np.zeros(len(x), dtype=int)
    amp = np.abs(x)
    for i, thr in enumerate(thresholds):
        labels[amp > thr] = i + 1

    n_classes = len(thresholds) + 1
    counts = {int(c): int(np.sum(labels == c)) for c in range(n_classes)}

    return DescriptiveResult(
        name="amplitude_classify",
        value=float(n_classes),
        extra={
            "labels": labels,
            "thresholds": thresholds,
            "counts": counts,
            "n_classes": n_classes,
        },
    )


ampcl = amplitude_classify


def cheatsheet() -> str:
    return "amplitude_classify({}) -> Classify signal segments by amplitude level."
