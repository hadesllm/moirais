# morie.fn — function file (hadesllm/morie)
"""Conditional probability."""


from ._containers import ESRes

_QUOTE = "Your focus determines your reality. -- Qui-Gon Jinn"


def conditional_prob(
    joint: float,
    marginal: float,
    **kwargs,
) -> ESRes:
    """
    Compute conditional probability P(A|B) = P(A and B) / P(B).

    .. math::

        P(A|B) = \\frac{P(A \\cap B)}{P(B)}

    :param joint: P(A ∩ B), joint probability.
    :param marginal: P(B), marginal probability of the conditioning event.
    :return: ESRes with conditional probability.
    :raises ValueError: If marginal is zero or inputs outside [0, 1].
    """
    if marginal <= 0:
        raise ValueError("marginal P(B) must be > 0.")
    if not 0 <= joint <= 1:
        raise ValueError("joint probability must be in [0, 1].")
    if joint > marginal:
        raise ValueError("joint cannot exceed marginal.")
    cond = joint / marginal
    return ESRes(
        measure="conditional_probability",
        estimate=float(cond),
        extra={"joint": joint, "marginal": marginal},
    )


condp = conditional_prob


def cheatsheet() -> str:
    return "conditional_prob({}) -> Conditional probability."
