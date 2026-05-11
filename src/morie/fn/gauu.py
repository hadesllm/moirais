# morie.fn — function file (hadesllm/morie)
"""Gaussian utility function for spatial voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def gaussian_utility(ideal, position, beta: float = 1.0, w: float = 1.0) -> DescriptiveResult:
    """Gaussian utility U(x) = exp(-0.5 * w * (x - z)^2).

    :param ideal: Voter ideal point.
    :param position: Candidate/policy position.
    :param beta: Overall scale.
    :param w: Width parameter.
    :return: DescriptiveResult with utility value.

    .. epigraph:: "It's over 9000!" -- Vegeta, Dragon Ball Z
    """
    import numpy as np

    ideal = np.asarray(ideal, dtype=float)
    position = np.asarray(position, dtype=float)
    diff = ideal - position
    utility = beta * np.exp(-0.5 * w * np.sum(diff**2))
    return DescriptiveResult(
        name="gaussian_utility",
        value=float(utility),
        extra={"ideal": ideal.tolist(), "position": position.tolist(), "beta": beta, "w": w},
    )


gauu = gaussian_utility


def cheatsheet() -> str:
    return "gaussian_utility({}) -> Gaussian utility function for spatial voting."
