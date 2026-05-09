# moirais.fn — function file (hadesllm/moirais)
"""Convert odds ratio to Pearson r via d."""

from .d2r import d_to_r
from .or2d import or_to_d


def or_to_r(or_val: float) -> float:
    """Convert odds ratio to Pearson r via d.

    Parameters
    ----------
    or_val : float

    Returns
    -------
    float
    """
    return d_to_r(or_to_d(or_val))


or2r = or_to_r


def cheatsheet() -> str:
    return "or_to_r({}) -> Convert odds ratio to Pearson r via d."
