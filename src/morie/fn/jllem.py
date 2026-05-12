# morie.fn -- function file (hadesllm/morie)
"""Johnson-Lindenstrauss lemma dimension bound."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def jl_lemma_bound(
    n: int,
    eps: float = 0.1,
) -> DescriptiveResult:
    """Compute the JL lemma lower bound on target dimension.

    The Johnson-Lindenstrauss lemma states that *n* points can be
    embedded into d >= O(log(n)/eps^2) dimensions while preserving
    all pairwise distances within (1 +/- eps).

    :param n: Number of points.
    :param eps: Distortion tolerance (0 < eps < 1).
    :return: DescriptiveResult with minimum dimension.
    :raises ValueError: If eps not in (0, 1) or n < 2.
    """
    if not 0 < eps < 1:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    d_min = int(np.ceil(8 * np.log(n) / (eps**2)))
    return DescriptiveResult(
        name="jl_lemma_bound",
        value=float(d_min),
        extra={
            "n": n,
            "eps": eps,
            "formula": "d >= 8 * ln(n) / eps^2",
        },
    )


def cheatsheet() -> str:
    return "jl_lemma_bound(n, eps) -> JL dimension lower bound d >= O(log(n)/eps^2)"


jllem = jl_lemma_bound
