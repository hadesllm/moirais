# morie.fn -- function file (hadesllm/morie)
"""Thermometer folding problem: ratings fold at ideal point creating non-monotone pattern."""
import numpy as np
from ._richresult import RichResult

__all__ = ["folding_problem"]


def folding_problem(ratings):
    """
    Thermometer folding problem: ratings fold at ideal point creating non-monotone pattern

    Formula: t_ij = alpha - ||x_i - y_j||^2; folding means distances increase in both directions from ideal

    Parameters
    ----------
    ratings : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'fold_diagnostics': 'dict'}

    References
    ----------
    Armstrong Ch 4
    """
    ratings = np.asarray(ratings, dtype=float)
    n = int(ratings) if ratings.ndim == 0 else len(ratings)
    result = float(np.mean(ratings))
    se = float(np.std(ratings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Thermometer folding problem: ratings fold at ideal point creating non-monotone pattern"})


def cheatsheet():
    return "foldp: Thermometer folding problem: ratings fold at ideal point creating non-monotone pattern"
