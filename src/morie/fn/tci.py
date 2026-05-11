# morie.fn — function file (hadesllm/morie)
"""t-based CI half-width."""

from scipy.stats import t as _t
def tci(s: float, n: int, conf: float = 0.95) -> float:
    """Half-width of a t-based confidence interval.

    margin = t_(1−α/2; n−1) × s / √n
    """
    if s <= 0 or n < 2 or not 0 < conf < 1:
        raise ValueError("invalid arguments.")
    crit = _t.ppf(0.5 + conf / 2, df=n - 1)
    return float(crit * s / (n ** 0.5))
