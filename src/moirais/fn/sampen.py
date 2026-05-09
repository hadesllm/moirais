# moirais.fn — function file (hadesllm/moirais)
"""Sample entropy."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def sample_entropy(
    x: np.ndarray,
    *,
    m: int = 2,
    r: float = 0.2,
) -> DescriptiveResult:
    """Sample entropy of a 1-D signal.

    :param x: 1-D input signal.
    :param m: Embedding dimension (default 2).
    :param r: Tolerance as fraction of std (default 0.2).
    :return: DescriptiveResult with SampEn in ``value``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    tol = r * np.std(x, ddof=1)
    if tol == 0 or n < m + 2:
        return DescriptiveResult(name="sample_entropy", value=float("nan"))

    def _count_matches(dim: int) -> int:
        count = 0
        templates = np.array([x[i : i + dim] for i in range(n - dim)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= tol:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)

    if b == 0:
        return DescriptiveResult(name="sample_entropy", value=float("inf"))

    se = -np.log(a / b)

    return DescriptiveResult(
        name="sample_entropy",
        value=float(se),
        extra={"m": m, "r": r, "tolerance": tol, "A": a, "B": b},
    )


sampen = sample_entropy


def cheatsheet() -> str:
    return "sample_entropy({}) -> Sample entropy."
