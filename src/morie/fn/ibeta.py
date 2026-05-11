# morie.fn — function file (hadesllm/morie)
"""Regularised incomplete Beta."""

from scipy.special import betainc
def ibeta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b).

    Equals the CDF of Beta(a, b) at x.
    """
    if not 0 <= x <= 1:
        raise ValueError("x must be in [0, 1].")
    return float(betainc(a, b, x))
