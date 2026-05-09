# moirais.fn — function file (hadesllm/moirais)
"""Cosine similarity via QJL projection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qjl_cosine_sim(
    x: np.ndarray,
    y: np.ndarray,
    d_proj: int = 128,
    seed: int = 42,
) -> DescriptiveResult:
    """Approximate cosine similarity using QJL sign-projection.

    Projects both vectors via a shared Rademacher matrix, then
    estimates cosine similarity from sign agreement.

    :param x: First vector.
    :param y: Second vector.
    :param d_proj: Projection dimension.
    :param seed: Random seed.
    :return: DescriptiveResult with approximate cosine similarity.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    d = len(x)
    rng = np.random.default_rng(seed)
    R = rng.choice([-1.0, 1.0], size=(d, d_proj))
    sx = np.sign(x @ R)
    sy = np.sign(y @ R)
    agreement = float(np.mean(sx == sy))
    cos_approx = float(np.cos(np.pi * (1 - agreement)))
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    cos_exact = float(np.dot(x, y) / (nx * ny)) if nx > 0 and ny > 0 else 0.0
    return DescriptiveResult(
        name="qjl_cosine_sim",
        value=cos_approx,
        extra={
            "exact": cos_exact,
            "agreement": agreement,
            "d_proj": d_proj,
            "error": abs(cos_approx - cos_exact),
        },
    )


def cheatsheet() -> str:
    return "qjl_cosine_sim(x, y, d_proj) -> cosine similarity via QJL"


qjlcs = qjl_cosine_sim
