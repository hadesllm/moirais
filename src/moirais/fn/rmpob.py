# moirais.fn — function file (hadesllm/moirais)
"""Random match probability for forensic genetics."""

__all__ = ["rmpob"]

import numpy as np

from ._containers import GenomicsResult


def rmpob(
    allele_freqs: list | np.ndarray,
    *,
    theta: float = 0.0,
) -> GenomicsResult:
    """Random match probability (RMP) for a forensic DNA profile.

    Computes the probability that a random unrelated individual
    matches the observed profile at all typed loci.

    Parameters
    ----------
    allele_freqs : list of tuples or array
        For each locus, a tuple (p_i, p_j) giving the population
        frequencies of the two observed alleles. For homozygotes,
        p_i == p_j.  Shape: list of length L, each element a 2-tuple.
    theta : float
        Population structure correction (Balding-Nichols theta/Fst).
        Default 0 (no correction).

    Returns
    -------
    GenomicsResult
        statistic = -log10(RMP), p_value = RMP,
        extra has 'per_locus_prob'.

    References
    ----------
    Balding, D. J., & Nichols, R. A. (1994). DNA profile match
        probability calculation: how to allow for population
        stratification, relatedness, database selection and single
        bands. Forensic Sci. Int., 64(2-3), 125-140.
    NRC II (1996). The Evaluation of Forensic DNA Evidence.
        National Academies Press.
    """
    locus_probs = []

    for locus_data in allele_freqs:
        pi, pj = float(locus_data[0]), float(locus_data[1])
        if pi < 0 or pj < 0 or pi > 1 or pj > 1:
            raise ValueError("Allele frequencies must be in [0, 1].")

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

        locus_probs.append(max(prob, 1e-30))

    rmp = float(np.prod(locus_probs))
    log10_rmp = -np.log10(max(rmp, 1e-300))

    return GenomicsResult(
        name="RMP",
        statistic=float(log10_rmp),
        p_value=rmp,
        n=len(allele_freqs),
        extra={"per_locus_prob": locus_probs, "theta": theta},
    )


def cheatsheet() -> str:
    return "rmpob(allele_freqs) -> Random match probability (forensic)."
