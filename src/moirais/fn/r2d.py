# moirais.fn — function file (hadesllm/moirais)
"""Convert Pearson r to Cohen's d."""

import math

import numpy as np


def r_to_d(r: float) -> float:
    """Convert Pearson r to Cohen's d.

    d = 2r / sqrt(1 - r^2)

    Parameters
    ----------
    r : float

    Returns
    -------
    float
    """
    return 2 * r / math.sqrt(1 - r**2) if abs(r) < 1 else np.inf * np.sign(r)


r2d = r_to_d


def cheatsheet() -> str:
    return "r_to_d({}) -> Convert Pearson r to Cohen's d."
