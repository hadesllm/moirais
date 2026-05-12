# morie.fn -- function file (hadesllm/morie)
"""Convert odds ratio to Cohen's d (Hasselblad and Hedges, 1995)."""

import math


def or_to_d(or_val: float) -> float:
    """Convert odds ratio to Cohen's d.

    d = log(OR) * sqrt(3) / pi

    Parameters
    ----------
    or_val : float
        Odds ratio.

    Returns
    -------
    float
    """
    return math.log(or_val) * math.sqrt(3) / math.pi if or_val > 0 else 0.0


or2d = or_to_d


def cheatsheet() -> str:
    return "or_to_d({}) -> Convert odds ratio to Cohen's d (Hasselblad and Hedges, 1995"
