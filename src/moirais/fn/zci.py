# moirais.fn — function file (hadesllm/moirais)
"""z-based confidence interval half-width."""

from scipy.stats import norm
def zci(sigma: float, n: int, conf: float = 0.95) -> float:
    """Half-width of a z-based confidence interval for the mean.

    margin = z_(1−α/2) × σ / √n
    """
    if sigma <= 0 or n < 1 or not 0 < conf < 1:
        raise ValueError("invalid arguments.")
    z = norm.ppf(0.5 + conf / 2)
    return float(z * sigma / (n ** 0.5))
