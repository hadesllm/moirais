# morie.fn -- function file (hadesllm/morie)
"""Hardy-Weinberg equilibrium chi-square test."""

import numpy as np
from scipy import stats

from ._containers import GenomicsResult


def hardy_weinberg_test(n_AA: int, n_Aa: int, n_aa: int, cdf=None) -> GenomicsResult:
    """Chi-square goodness-of-fit test for Hardy-Weinberg equilibrium.

    Given observed genotype counts (AA, Aa, aa), estimate allele
    frequencies p = freq(A) and q = 1 - p, compute expected counts
    under HWE (p^2, 2pq, q^2)*N, and perform a chi-square test
    with 1 degree of freedom.

    Parameters
    ----------
    n_AA : int
        Observed count of homozygous dominant (AA).
    n_Aa : int
        Observed count of heterozygous (Aa).
    n_aa : int
        Observed count of homozygous recessive (aa).

    Returns
    -------
    GenomicsResult
        name="Hardy-Weinberg", statistic=chi2, p_value=p.

    Raises
    ------
    ValueError
        If any count is negative or total is zero.

    References
    ----------
    Hardy, G. H. (1908). Mendelian proportions in a mixed population.
        Science, 28(706), 49-50.
    Weinberg, W. (1908). Uber den Nachweis der Vererbung beim Menschen.
        Jahreshefte des Vereins fur Vaterlandische Naturkunde in
        Wurttemberg, 64, 368-382.
    """
    if n_AA < 0 or n_Aa < 0 or n_aa < 0:
        raise ValueError("Genotype counts must be non-negative.")
    n = n_AA + n_Aa + n_aa
    if n == 0:
        raise ValueError("Total sample size must be > 0.")

    p = (2 * n_AA + n_Aa) / (2 * n)
    q = 1.0 - p

    exp_AA = p**2 * n
    exp_Aa = 2 * p * q * n
    exp_aa = q**2 * n

    observed = np.array([n_AA, n_Aa, n_aa], dtype=float)
    expected = np.array([exp_AA, exp_Aa, exp_aa])

    mask = expected > 0
    chi2 = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    p_value = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return GenomicsResult(
        name="Hardy-Weinberg",
        statistic=chi2,
        p_value=p_value,
        n=n,
        extra={"p_allele": p, "q_allele": q, "expected": {"AA": exp_AA, "Aa": exp_Aa, "aa": exp_aa}},
    )


hw = hardy_weinberg_test


def cheatsheet() -> str:
    return "hardy_weinberg_test({}) -> Hardy-Weinberg equilibrium chi-square test."
