# morie.fn — function file (hadesllm/morie)
"""Confusion matrix and classification metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cfmat_fn(y_true: np.ndarray, y_pred: np.ndarray) -> DescriptiveResult:
    """Compute confusion matrix and derived classification metrics.

    :param y_true: True class labels.
    :param y_pred: Predicted class labels.
    :return: DescriptiveResult with accuracy and full metrics dict.
    """
    from morie._classify import confusion_matrix_metrics

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = confusion_matrix_metrics(y_true, y_pred)
    return DescriptiveResult(
        name="confusion_matrix",
        value=metrics["accuracy"],
        extra=metrics,
    )


cfmat = cfmat_fn


def cheatsheet() -> str:
    return "cfmat_fn({}) -> Confusion matrix and classification metrics."
