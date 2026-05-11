# morie.fn — function file (hadesllm/morie)
"""z critical value."""

from scipy.stats import norm
def zcrit(alpha: float = 0.05, two_sided: bool = True) -> float:
    """Two-sided z critical value at significance level α."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")
    if two_sided:
        return float(norm.ppf(1 - alpha / 2))
    return float(norm.ppf(1 - alpha))
