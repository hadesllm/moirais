# moirais.fn — function file (hadesllm/moirais)
"""Quadratic utility function for spatial voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def quadratic_utility(ideal, position, beta: float = 1.0) -> DescriptiveResult:
    """Quadratic utility U(x) = -beta * (x - z)^2.

    :param ideal: Voter ideal point (scalar or array).
    :param position: Candidate/policy position (scalar or array).
    :param beta: Salience weight (default 1.0).
    :return: DescriptiveResult with utility value.

    .. epigraph:: "Fear is the mind-killer." -- Bene Gesserit, Dune
    """
    import numpy as np

    ideal = np.asarray(ideal, dtype=float)
    position = np.asarray(position, dtype=float)
    diff = ideal - position
    utility = -beta * np.sum(diff**2)
    return DescriptiveResult(
        name="quadratic_utility",
        value=float(utility),
        extra={"ideal": ideal.tolist(), "position": position.tolist(), "beta": beta},
    )


quadu = quadratic_utility


def cheatsheet() -> str:
    return "quadratic_utility({}) -> Quadratic utility function for spatial voting."
