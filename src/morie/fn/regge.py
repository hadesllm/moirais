# morie.fn -- function file (hadesllm/morie)
"""Regge trajectory alpha(s) = alpha_0 + alpha_prime * s."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def regge_trajectory(
    alpha0: float = 0.5,
    alpha_prime: float = 0.9,
    s: float | np.ndarray = 1.0,
) -> DescriptiveResult:
    """Compute the Regge trajectory alpha(s) = alpha_0 + alpha' * s.

    Linear Regge trajectories relate spin J to mass squared:

    .. math::

        \\alpha(s) = \\alpha_0 + \\alpha' s

    :param alpha0: Intercept of the trajectory.
    :param alpha_prime: Slope (inverse string tension).
    :param s: Mandelstam variable or mass squared values.
    :return: DescriptiveResult with trajectory values.
    """
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    alpha = alpha0 + alpha_prime * s_arr
    return DescriptiveResult(
        name="regge_trajectory",
        value=float(alpha[0]) if alpha.size == 1 else None,
        extra={
            "alpha": alpha,
            "alpha0": alpha0,
            "alpha_prime": alpha_prime,
            "s": s_arr,
        },
    )


def cheatsheet() -> str:
    return "regge_trajectory(alpha0, alpha_prime, s) -> Regge trajectory alpha(s)"


regge = regge_trajectory
