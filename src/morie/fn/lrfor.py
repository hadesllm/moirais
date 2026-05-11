# morie.fn — function file (hadesllm/morie)
"""Likelihood ratio for forensic genetics."""

__all__ = ["lrfor"]

import numpy as np

from ._containers import GenomicsResult


def lrfor(
    allele_freqs: list | np.ndarray,
    *,
    theta: float = 0.0,
    hypothesis: str = "prosecution",
) -> GenomicsResult:
    """Likelihood ratio (LR) for forensic DNA evidence.

    Computes the LR comparing prosecution (suspect is source) to
    defense (random person is source) hypotheses.

    LR = 1 / P(profile | H_d) = 1 / RMP

    Parameters
    ----------
    allele_freqs : list of tuples
        For each locus, (p_i, p_j) allele frequencies.
        Homozygotes: p_i == p_j.
    theta : float
        Balding-Nichols population structure correction.
    hypothesis : {'prosecution', 'defense'}
        Which hypothesis is numerator.  Default 'prosecution'
        gives LR = 1/RMP.

    Returns
    -------
    GenomicsResult
        statistic = log10(LR), extra has 'LR', 'per_locus_lr'.

    References
    ----------
    Evett, I. W., & Weir, B. S. (1998). Interpreting DNA Evidence:
        Statistical Genetics for Forensic Scientists. Sinauer.
    """
    per_locus_lr = []

    for locus_data in allele_freqs:
        pi, pj = float(locus_data[0]), float(locus_data[1])
        if pi < 0 or pj < 0:
            raise ValueError("Allele frequencies must be non-negative.")

        if theta > 0:
            if abs(pi - pj) < 1e-10:
                prob = (
                    (2 * theta + (1 - theta) * pi)
                    * (3 * theta + (1 - theta) * pi)
                    / ((1 + theta) * (1 + 2 * theta))
                )
            else:
                prob = (
                    2 * (theta + (1 - theta) * pi)
                    * (theta + (1 - theta) * pj)
                    / ((1 + theta) * (1 + 2 * theta))
                )
        else:
            if abs(pi - pj) < 1e-10:
                prob = pi ** 2
            else:
                prob = 2 * pi * pj

        per_locus_lr.append(1.0 / max(prob, 1e-30))

    lr = float(np.prod(per_locus_lr))
    if hypothesis == "defense":
        lr = 1.0 / max(lr, 1e-30)
        per_locus_lr = [1.0 / x for x in per_locus_lr]

    log10_lr = np.log10(max(lr, 1e-300))

    return GenomicsResult(
        name="ForensicLR",
        statistic=float(log10_lr),
        n=len(allele_freqs),
        extra={
            "LR": lr,
            "per_locus_lr": per_locus_lr,
            "theta": theta,
            "hypothesis": hypothesis,
        },
    )


def cheatsheet() -> str:
    return "lrfor(allele_freqs) -> Forensic likelihood ratio."
