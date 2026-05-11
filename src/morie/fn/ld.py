# morie.fn — function file (hadesllm/morie)
"""Linkage disequilibrium (r-squared)."""

import numpy as np

from ._containers import GenomicsResult


def linkage_disequilibrium(
    locus_a: np.ndarray | list,
    locus_b: np.ndarray | list,
) -> GenomicsResult:
    """Linkage disequilibrium measured as r-squared.

    r^2 = D^2 / (p_A * (1 - p_A) * p_B * (1 - p_B))

    where D = freq(AB) - p_A * p_B and alleles are coded 0/1.

    Parameters
    ----------
    locus_a : array-like of {0, 1}
        Allele states at locus A for each haplotype.
    locus_b : array-like of {0, 1}
        Allele states at locus B for each haplotype.

    Returns
    -------
    GenomicsResult
        name="LD_r2", statistic=r^2 in [0, 1].

    Raises
    ------
    ValueError
        If arrays differ in length or contain values other than 0/1.

    References
    ----------
    Lewontin, R. C. (1964). The interaction of selection and linkage.
        I. General considerations; heterotic models. Genetics, 49(1), 49-67.
    Hill, W. G., & Robertson, A. (1968). Linkage disequilibrium in finite
        populations. Theoretical and Applied Genetics, 38(6), 226-231.
    """
    a = np.asarray(locus_a, dtype=float)
    b = np.asarray(locus_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Locus arrays must have the same length.")
    if len(a) == 0:
        raise ValueError("Arrays must not be empty.")
    if not np.all(np.isin(a, [0, 1])):
        raise ValueError("locus_a must contain only 0 and 1.")
    if not np.all(np.isin(b, [0, 1])):
        raise ValueError("locus_b must contain only 0 and 1.")

    n = len(a)
    pA = np.mean(a)
    pB = np.mean(b)

    denom = pA * (1.0 - pA) * pB * (1.0 - pB)
    if denom == 0:
        return GenomicsResult(name="LD_r2", statistic=0.0, n=n, extra={"D": 0.0, "pA": pA, "pB": pB})

    freq_AB = np.mean(a * b)
    D = freq_AB - pA * pB
    r2 = (D * D) / denom

    return GenomicsResult(
        name="LD_r2",
        statistic=float(r2),
        n=n,
        extra={"D": float(D), "pA": float(pA), "pB": float(pB)},
    )


ld = linkage_disequilibrium


def cheatsheet() -> str:
    return "linkage_disequilibrium({}) -> Linkage disequilibrium (r-squared)."
