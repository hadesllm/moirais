# morie.fn -- function file (hadesllm/morie)
"""F critical value."""

from scipy.stats import f as _f
def fcrit(df1: int, df2: int, alpha: float = 0.05) -> float:
    """Upper-tail F critical value (df1, df2, α)."""
    if df1 < 1 or df2 < 1 or not 0 < alpha < 1:
        raise ValueError("invalid args.")
    return float(_f.ppf(1 - alpha, df1, df2))
