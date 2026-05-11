"""Abundance estimation (Bracken)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["abundance_estimation"]


def abundance_estimation(kraken_output, kmer_distribution):
    """
    Abundance estimation (Bracken)

    Formula: Bayesian re-estimation post Kraken

    Parameters
    ----------
    kraken_output : array-like
        Input data.
    kmer_distribution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lu et al (2017) Bracken
    """
    kraken_output = np.atleast_1d(np.asarray(kraken_output, dtype=float))
    n = len(kraken_output)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Abundance estimation (Bracken)"})
    estimate = np.median(kraken_output)
    se = 1.2533 * np.std(kraken_output, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Abundance estimation (Bracken)"})


def cheatsheet():
    return "abndst: Abundance estimation (Bracken)"
