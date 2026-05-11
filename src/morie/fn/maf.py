# morie.fn — function file (hadesllm/morie)
"""Minor allele frequency from genotype data."""

import numpy as np

from ._containers import ESRes


def minor_allele_frequency(
    genotypes: np.ndarray | list,
) -> ESRes:
    """Minor allele frequency (MAF) from genotype coding (0, 1, 2).

    p = sum(genotypes) / (2 * N)
    MAF = min(p, 1 - p)

    where 0 = homozygous reference, 1 = heterozygous,
    2 = homozygous alternate.

    Parameters
    ----------
    genotypes : array-like of {0, 1, 2}
        Genotype dosage for each individual.

    Returns
    -------
    ESRes
        measure="MAF", estimate=minor allele frequency in [0, 0.5].

    Raises
    ------
    ValueError
        If genotypes is empty or contains values outside {0, 1, 2}.

    References
    ----------
    LaFramboise, T. (2009). Single nucleotide polymorphism arrays:
        a decade of biological, computational and technological advances.
        Nucleic Acids Research, 37(13), 4181-4193.
    """
    g = np.asarray(genotypes, dtype=float)
    if len(g) == 0:
        raise ValueError("Genotypes array must not be empty.")
    if not np.all(np.isin(g, [0, 1, 2])):
        raise ValueError("Genotypes must contain only 0, 1, or 2.")

    n = len(g)
    p = float(np.sum(g) / (2 * n))
    maf_val = min(p, 1.0 - p)

    return ESRes(
        measure="MAF",
        estimate=maf_val,
        n=n,
        extra={"p": p, "q": 1.0 - p},
    )


maf = minor_allele_frequency


def cheatsheet() -> str:
    return "minor_allele_frequency({}) -> Minor allele frequency from genotype data."
