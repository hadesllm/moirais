# morie.fn -- function file (hadesllm/morie)
"""Chi-squared critical value."""

from scipy.stats import chi2
def chcrit(df: int, alpha: float = 0.05) -> float:
    """Upper-tail χ² critical value (df, α)."""
    if df < 1 or not 0 < alpha < 1:
        raise ValueError("invalid df or alpha.")
    return float(chi2.ppf(1 - alpha, df))
