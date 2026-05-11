# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bhattacharyya divergence between two distributions."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bhatt_fn(X1: np.ndarray, X2: np.ndarray) -> DescriptiveResult:
    """Compute Bhattacharyya divergence between two sample sets.

    :param X1: Samples from class 1 (samples x features).
    :param X2: Samples from class 2 (samples x features).
    :return: DescriptiveResult with divergence value.
    """
    from morie._classify import bhattacharyya_divergence

    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    divergence = bhattacharyya_divergence(X1, X2)
    return DescriptiveResult(
        name="bhattacharyya",
        value=divergence,
        extra={"divergence": divergence},
    )


bhatt = bhatt_fn


def cheatsheet() -> str:
    return "bhatt_fn({}) -> Bhattacharyya divergence between two distributions."
