# morie.fn -- function file (hadesllm/morie)
"""QJL projection matrix generator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qjl_matrix(
    d_in: int,
    d_out: int,
    seed: int = 42,
) -> DescriptiveResult:
    """Generate a QJL Rademacher projection matrix.

    Entries are +/-1 with equal probability, scaled by 1/sqrt(d_out).

    :param d_in: Input dimensionality.
    :param d_out: Output (projected) dimensionality.
    :param seed: Random seed.
    :return: DescriptiveResult with projection matrix.
    """
    rng = np.random.default_rng(seed)
    R = rng.choice([-1.0, 1.0], size=(d_in, d_out)) / np.sqrt(d_out)
    return DescriptiveResult(
        name="qjl_matrix",
        value=float(d_in * d_out),
        extra={
            "matrix": R,
            "d_in": d_in,
            "d_out": d_out,
            "expected_norm": 1.0,
        },
    )


def cheatsheet() -> str:
    return "qjl_matrix(d_in, d_out, seed) -> generate QJL projection matrix"


qjlmt = qjl_matrix
