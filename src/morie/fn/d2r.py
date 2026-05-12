# morie.fn -- function file (hadesllm/morie)
"""Convert Cohen's d to Pearson r."""

import math


def d_to_r(d: float, n1: int | None = None, n2: int | None = None) -> float:
    """Convert Cohen's d to Pearson r.

    r = d / sqrt(d^2 + a) where a = (n1+n2)^2/(n1*n2) or 4 if unknown.

    Parameters
    ----------
    d : float
    n1, n2 : int or None

    Returns
    -------
    float
    """
    if n1 is not None and n2 is not None:
        a = (n1 + n2) ** 2 / (n1 * n2)
    else:
        a = 4.0
    return d / math.sqrt(d**2 + a)


d2r = d_to_r


def cheatsheet() -> str:
    return "d_to_r({}) -> Convert Cohen's d to Pearson r."
