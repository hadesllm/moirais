# morie.fn -- function file (hadesllm/morie)
"""Fisher Linear Discriminant Analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def flda_fn(X: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """Compute Fisher LDA projection for two-class discrimination.

    :param X: 2-D array (samples x features).
    :param y: 1-D class labels.
    :return: DescriptiveResult with threshold and weights/means.
    """
    from morie._classify import fisher_lda

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    w, means, threshold = fisher_lda(X, y)
    return DescriptiveResult(
        name="fisher_lda",
        value=threshold,
        extra={"weights": w, "means": means, "threshold": threshold},
    )


flda = flda_fn


def cheatsheet() -> str:
    return "flda_fn({}) -> Fisher Linear Discriminant Analysis."
