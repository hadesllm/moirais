# morie.fn -- function file (hadesllm/morie)
"""Linear (city-block) utility function for spatial voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def linear_utility(ideal, position) -> DescriptiveResult:
    """Linear utility U(x) = -|x - z|.

    :param ideal: Voter ideal point.
    :param position: Candidate/policy position.
    :return: DescriptiveResult with utility value.

    .. epigraph:: "Believe it!" -- Naruto Uzumaki, Naruto
    """
    import numpy as np

    ideal = np.asarray(ideal, dtype=float)
    position = np.asarray(position, dtype=float)
    utility = -np.sum(np.abs(ideal - position))
    return DescriptiveResult(
        name="linear_utility",
        value=float(utility),
        extra={"ideal": ideal.tolist(), "position": position.tolist()},
    )


linru = linear_utility


def cheatsheet() -> str:
    return "linear_utility({}) -> Linear (city-block) utility function for spatial voting."
