# moirais.fn — function file (hadesllm/moirais)
"""t critical value."""

from scipy.stats import t as _t
def tcrit(df: int, alpha: float = 0.05, two_sided: bool = True) -> float:
    """Two-sided t critical value at significance α with df."""
    if df < 1 or not 0 < alpha < 1:
        raise ValueError("invalid df or alpha.")
    if two_sided:
        return float(_t.ppf(1 - alpha / 2, df))
    return float(_t.ppf(1 - alpha, df))
