"""Levy flight random walk. 'Surprise!' -- Skywarp"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def levy_flight(
    n_steps: int = 1000,
    *,
    alpha: float = 1.5,
    dim: int = 2,
    seed: int = 42,
) -> DescriptiveResult:
    """Simulate a Levy flight random walk.

    Step lengths are drawn from a Levy stable distribution with
    stability parameter alpha. For alpha=2 this reduces to Brownian motion.

    Parameters
    ----------
    n_steps : int
        Number of steps.
    alpha : float
        Levy stability parameter (0 < alpha <= 2).
    dim : int
        Spatial dimension.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = trajectory (n_steps+1, dim) and
        ``extra`` containing step statistics.
    """
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    rng = np.random.default_rng(seed)

    if alpha == 2.0:
        steps = rng.standard_normal((n_steps, dim))
    else:
        import math

        sigma = (
            math.gamma(1 + alpha)
            * np.sin(np.pi * alpha / 2)
            / (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))
        ) ** (1 / alpha)
        u = rng.normal(0, sigma, (n_steps, dim))
        v = np.abs(rng.standard_normal((n_steps, dim)))
        steps = u / (v ** (1 / alpha))

    trajectory = np.zeros((n_steps + 1, dim))
    trajectory[1:] = np.cumsum(steps, axis=0)

    step_lengths = np.linalg.norm(steps, axis=1)
    displacement = np.linalg.norm(trajectory[-1])

    return DescriptiveResult(
        name="levy_flight",
        value=trajectory,
        extra={
            "alpha": alpha,
            "dim": dim,
            "n_steps": n_steps,
            "mean_step": float(step_lengths.mean()),
            "max_step": float(step_lengths.max()),
            "displacement": float(displacement),
        },
    )


skwrp = levy_flight


def cheatsheet() -> str:
    return "levy_flight({}) -> Levy flight random walk. 'Surprise!' -- Skywarp"
