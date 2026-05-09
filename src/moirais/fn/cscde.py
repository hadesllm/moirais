# moirais.fn — function file (hadesllm/moirais)
"""Cascade (sequential rejection) classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "If you strike me down I shall become more powerful."


def cascade_classify(X, classifiers, thresholds, **kwargs) -> DescriptiveResult:
    """Cascade classifier with sequential rejection stages.

    Each stage applies a classifier; samples above the threshold are
    accepted (classified positive), the rest pass to the next stage.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Input features.
    classifiers : list of callables
        Each takes X (n, p) and returns scores (n,).
    thresholds : list of float
        Decision threshold per stage.

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    predictions = np.full(n, -1)
    stage_assigned = np.full(n, -1, dtype=int)
    remaining = np.arange(n)

    for s, (clf, thresh) in enumerate(zip(classifiers, thresholds)):
        if len(remaining) == 0:
            break
        scores = np.asarray(clf(X[remaining]), dtype=float)
        accepted = scores >= thresh
        predictions[remaining[accepted]] = 1
        stage_assigned[remaining[accepted]] = s
        remaining = remaining[~accepted]

    predictions[remaining] = 0
    stage_assigned[remaining] = len(classifiers)

    n_accepted = int(np.sum(predictions == 1))
    n_rejected = int(np.sum(predictions == 0))

    return DescriptiveResult(
        name="cascade_classify",
        value=float(n_accepted / n) if n > 0 else 0.0,
        extra={
            "predictions": predictions,
            "stage_assigned": stage_assigned,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "n_stages": len(classifiers),
        },
    )


cscde = cascade_classify


def cheatsheet() -> str:
    return "cascade_classify({}) -> Cascade (sequential rejection) classifier."
