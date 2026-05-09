# moirais.fn — function file (hadesllm/moirais)
"""Regularised lower incomplete gamma."""

from scipy.special import gammainc
def igamma(a: float, x: float) -> float:
    """Regularised lower incomplete gamma function P(a, x).

    Equals CDF of Gamma(a, 1) at x.
    """
    if a <= 0 or x < 0:
        raise ValueError("require a>0 and x≥0.")
    return float(gammainc(a, x))
