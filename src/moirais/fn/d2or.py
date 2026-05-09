# moirais.fn — function file (hadesllm/moirais)
"""Convert Cohen's d to odds ratio."""

import math


def d_to_or(d: float) -> float:
    """Convert Cohen's d to an odds ratio.

    OR = exp(d * pi / sqrt(3))

    Parameters
    ----------
    d : float

    Returns
    -------
    float
    """
    return math.exp(d * math.pi / math.sqrt(3))


d2or = d_to_or


def cheatsheet() -> str:
    return "d_to_or({}) -> Convert Cohen's d to odds ratio."
