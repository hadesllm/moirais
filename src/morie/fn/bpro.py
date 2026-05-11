# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Brownian bridge process simulation."""

from __future__ import annotations

import numpy as np

__all__ = ["bpro"]


def bpro(
    n_points: int = 1000,
    *,
    seed: int | None = None,
) -> dict:
    r"""
    Simulate a Brownian bridge process on [0, 1].

    The Brownian bridge :math:`B(t) = W(t) - t W(1)` where :math:`W` is
    standard Brownian motion.

    :param n_points: Number of discretization points. Default 1000.
    :param seed: Random seed for reproducibility.
    :return: Dict with keys ``t``, ``bridge``, ``n_points``.
    :raises ValueError: If n_points < 2.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 5. Springer.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_points)
    dt = t[1] - t[0]

    increments = rng.standard_normal(n_points - 1) * np.sqrt(dt)
    w = np.zeros(n_points)
    w[1:] = np.cumsum(increments)

    bridge = w - t * w[-1]

    return {
        "t": t,
        "bridge": bridge,
        "n_points": n_points,
    }


def cheatsheet() -> str:
    return "bpro({n_points}) -> Brownian bridge process simulation."
