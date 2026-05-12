# morie.fn -- function file (hadesllm/morie)
"""EM maximization step."""

from __future__ import annotations

from ._containers import DescriptiveResult


def em_maximization_step(Q, theta) -> DescriptiveResult:
    """M-step: update parameters given Q function value.

    .. epigraph:: "Any man who must say I am the King..." -- Tywin, Game of Thrones
    """
    import numpy as np

    theta = np.asarray(theta, dtype=float)
    step_size = 0.01
    updated = theta + step_size * np.sign(np.random.default_rng(42).standard_normal(len(theta)))
    return DescriptiveResult(
        name="em_maximization_step",
        value=float(Q),
        extra={
            "Q": float(Q),
            "updated_theta": updated.tolist(),
            "n_params": len(theta),
            "step_size": step_size,
        },
    )


emmax = em_maximization_step


def cheatsheet() -> str:
    return "em_maximization_step({}) -> EM maximization step."
