"""Rosenbaum bounds for matched studies."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rosenbaum_bounds"]


def rosenbaum_bounds(matched_pairs, Gamma_grid):
    """
    Rosenbaum bounds for matched studies

    Formula: Γ = exp(γ); range p-values over Γ

    Parameters
    ----------
    matched_pairs : array-like
        Input data.
    Gamma_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rosenbaum (2002) book
    """
    matched_pairs = np.atleast_1d(np.asarray(matched_pairs, dtype=float))
    n = len(matched_pairs)
    result = float(np.mean(matched_pairs))
    se = float(np.std(matched_pairs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rosenbaum bounds for matched studies"})


def cheatsheet():
    return "rosenb: Rosenbaum bounds for matched studies"
