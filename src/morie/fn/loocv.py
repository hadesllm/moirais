# morie.fn -- function file (hadesllm/morie)
"""Leave-One-Out Cross-Validation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def loocv_fn(X: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """Evaluate classifier accuracy using leave-one-out cross-validation.

    :param X: Feature matrix (samples x features).
    :param y: Class labels.
    :return: DescriptiveResult with LOOCV accuracy.
    """
    from morie._classify import loocv

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    accuracy = loocv(X, y)
    return DescriptiveResult(
        name="loocv",
        value=accuracy,
        extra={"accuracy": accuracy},
    )


loocv_fn_alias = loocv_fn


def cheatsheet() -> str:
    return "loocv_fn({}) -> Leave-One-Out Cross-Validation."
