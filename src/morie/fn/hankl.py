# morie.fn — function file (hadesllm/morie)
"""Construct Hankel matrix from signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def hankel_matrix_fn(x: np.ndarray, L: int) -> DescriptiveResult:
    """Construct a Hankel (trajectory) matrix from a 1-D signal.

    The Hankel matrix has dimensions :math:`L \\times (N - L + 1)` where
    each column is a lagged copy of the signal.

    :param x: 1-D input signal.
    :param L: Number of rows (window length).
    :return: DescriptiveResult with Hankel matrix.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if L < 1 or L > N:
        raise ValueError("L must be between 1 and len(x)")
    K = N - L + 1
    H = np.zeros((L, K))
    for i in range(K):
        H[:, i] = x[i : i + L]
    return DescriptiveResult(
        name="hankel_matrix",
        value=None,
        extra={"matrix": H, "L": L, "K": K, "shape": (L, K)},
    )


hankl = hankel_matrix_fn


def cheatsheet() -> str:
    return "hankel_matrix_fn({}) -> Construct Hankel matrix from signal."
