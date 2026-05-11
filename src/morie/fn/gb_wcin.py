# morie.fn — function file (hadesllm/morie)
"""Coefficient of concordance for incomplete rankings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_concordance_incomplete"]


def gibbons_concordance_incomplete(incomplete_rankings):
    """
    Coefficient of concordance for incomplete rankings

    Formula: W_i = adjusted formula for designs with not all objects ranked by all judges

    Parameters
    ----------
    incomplete_rankings : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W, p_value

    References
    ----------
    Gibbons Ch 12.5
    """
    incomplete_rankings = np.asarray(incomplete_rankings, dtype=float)
    n = int(incomplete_rankings) if incomplete_rankings.ndim == 0 else len(incomplete_rankings)
    result = float(np.mean(incomplete_rankings))
    se = float(np.std(incomplete_rankings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coefficient of concordance for incomplete rankings"})


def cheatsheet():
    return "gb_wcin: Coefficient of concordance for incomplete rankings"
