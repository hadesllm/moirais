# moirais.fn — function file (hadesllm/moirais)
"""Convert Pearson r to odds ratio via d."""

from .d2or import d_to_or
from .r2d import r_to_d


def r_to_or(r: float) -> float:
    """Convert Pearson r to odds ratio via d.

    Parameters
    ----------
    r : float

    Returns
    -------
    float
    """
    return d_to_or(r_to_d(r))


r2or = r_to_or


def cheatsheet() -> str:
    return "r_to_or({}) -> Convert Pearson r to odds ratio via d."
