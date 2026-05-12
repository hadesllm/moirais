# morie.fn -- function file (hadesllm/morie)
"""Kinship likelihood ratio for forensic genetics."""

__all__ = ["kinlr"]

import numpy as np

from ._containers import GenomicsResult


def kinlr(
    genotype_a: list,
    genotype_b: list,
    allele_freqs: list,
    *,
    relationship: str = "parent_child",
    theta: float = 0.0,
) -> GenomicsResult:
    """Likelihood ratio for kinship testing.

    Computes the LR comparing a claimed relationship (H_p) to
    the hypothesis that the two individuals are unrelated (H_d).

    Parameters
    ----------
    genotype_a : list of tuples
        Genotype of individual A at each locus, e.g. [(a1, a2), ...].
    genotype_b : list of tuples
        Genotype of individual B at each locus.
    allele_freqs : list of dicts
        For each locus, dict mapping allele -> frequency.
    relationship : str
        One of 'parent_child', 'full_sibling', 'half_sibling',
        'grandparent', 'uncle'.
    theta : float
        Population structure correction.

    Returns
    -------
    GenomicsResult
        statistic = log10(LR), extra has 'LR', 'per_locus_lr',
        'relationship'.

    References
    ----------
    Brenner, C. H. (1997). Symbolic kinship program. Genetics,
        145(2), 535-542.
    Gjertson, D. W., et al. (2007). ISFG: Recommendations on
        biostatistics in paternity testing. Forensic Sci. Int.:
        Genetics, 1(3-4), 223-231.
    """
    n_loci = len(genotype_a)
    if len(genotype_b) != n_loci or len(allele_freqs) != n_loci:
        raise ValueError("All inputs must have the same number of loci.")

    ibd_coeffs = {
        "parent_child": (0.0, 1.0, 0.0),
        "full_sibling": (0.25, 0.5, 0.25),
        "half_sibling": (0.5, 0.5, 0.0),
        "grandparent": (0.5, 0.5, 0.0),
        "uncle": (0.5, 0.5, 0.0),
        "unrelated": (1.0, 0.0, 0.0),
    }

    if relationship not in ibd_coeffs:
        raise ValueError(f"Unknown relationship: {relationship}")

    k0, k1, k2 = ibd_coeffs[relationship]

    per_locus_lr = []

    for loc in range(n_loci):
        a1, a2 = genotype_a[loc]
        b1, b2 = genotype_b[loc]
        freq = allele_freqs[loc]

        def _f(allele):
            return max(freq.get(allele, 1e-4), 1e-6)

        pa1, pa2 = _f(a1), _f(a2)
        pb1, pb2 = _f(b1), _f(b2)

        if a1 == a2:
            p_geno_a = pa1 ** 2
        else:
            p_geno_a = 2 * pa1 * pa2

        if b1 == b2:
            p_geno_b = pb1 ** 2
        else:
            p_geno_b = 2 * pb1 * pb2

        p_ibd0 = p_geno_b

        shared = set()
        geno_a_set = {a1, a2}
        geno_b_set = {b1, b2}
        shared = geno_a_set & geno_b_set

        if len(shared) == 0:
            p_ibd1 = 0.0
            p_ibd2 = 0.0
        elif geno_a_set == geno_b_set:
            if a1 == a2:
                p_ibd1 = pa1
                p_ibd2 = 1.0
            else:
                p_ibd1 = pa1 + pa2
                p_ibd2 = 1.0
        else:
            common = list(shared)[0]
            p_ibd1 = _f(common)
            p_ibd2 = 0.0

        p_hp = k0 * p_ibd0 + k1 * p_ibd1 + k2 * p_ibd2
        p_hd = p_geno_b

        lr_loc = p_hp / max(p_hd, 1e-30)
        per_locus_lr.append(float(lr_loc))

    combined_lr = float(np.prod(per_locus_lr))
    log10_lr = np.log10(max(combined_lr, 1e-300))

    return GenomicsResult(
        name="KinshipLR",
        statistic=float(log10_lr),
        n=n_loci,
        extra={
            "LR": combined_lr,
            "per_locus_lr": per_locus_lr,
            "relationship": relationship,
            "ibd_coefficients": {"k0": k0, "k1": k1, "k2": k2},
        },
    )


def cheatsheet() -> str:
    return "kinlr(geno_a, geno_b, freqs) -> Kinship likelihood ratio."
