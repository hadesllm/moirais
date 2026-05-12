# morie.fn -- function file (hadesllm/morie)
"""Genomic inflation factor (lambda GC)."""

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def genomic_inflation(chi2_stats: np.ndarray | None = None, p_values: np.ndarray | None = None) -> ESRes:
    r"""
    Compute the genomic inflation factor (lambda GC).

    .. math::

        \\lambda_{GC} = \\frac{\\text{median}(\\chi^2_{obs})}{0.4549}

    :param chi2_stats: Array of chi-squared statistics (1 df).
    :param p_values: Array of p-values (converted to chi2 internally).
    :return: ESRes with lambda as estimate.
    :raises ValueError: If neither chi2_stats nor p_values provided.

    References
    ----------
    Devlin B, Roeder K (1999). Genomic control for association studies.
    Biometrics, 55(4), 997-1004.
    """
    if chi2_stats is not None:
        chi2 = np.asarray(chi2_stats, dtype=np.float64)
    elif p_values is not None:
        pv = np.asarray(p_values, dtype=np.float64)
        chi2 = sp_stats.chi2.ppf(1 - pv, df=1)
    else:
        raise ValueError("Provide either chi2_stats or p_values.")
    expected_median = float(sp_stats.chi2.ppf(0.5, df=1))
    lambda_gc = float(np.median(chi2) / expected_median)
    return ESRes(
        measure="genomic_inflation",
        estimate=lambda_gc,
        extra={"median_chi2": float(np.median(chi2)), "expected_median": expected_median, "n_tests": len(chi2)},
    )


lambd = genomic_inflation


def cheatsheet() -> str:
    return "genomic_inflation({}) -> Genomic inflation factor (lambda GC)."
