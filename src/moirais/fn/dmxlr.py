# moirais.fn — function file (hadesllm/moirais)
"""DNA mixture likelihood ratio."""

__all__ = ["dmxlr"]

from itertools import product as _product

import numpy as np

from ._containers import GenomicsResult


def dmxlr(
    mixture_alleles: list,
    suspect_alleles: list,
    allele_freqs: list,
    *,
    n_contributors: int = 2,
    theta: float = 0.0,
) -> GenomicsResult:
    """Likelihood ratio for DNA mixture interpretation.

    Computes the LR for a suspect being a contributor to a DNA
    mixture vs the alternative that the mixture comes from
    unknown individuals.

    Parameters
    ----------
    mixture_alleles : list of sets or lists
        Alleles detected in the mixture at each locus.
    suspect_alleles : list of tuples
        Suspect's genotype at each locus, (a1, a2).
    allele_freqs : list of dicts
        For each locus, a dict mapping allele -> frequency.
    n_contributors : int
        Assumed number of contributors to the mixture.
    theta : float
        Population structure correction.

    Returns
    -------
    GenomicsResult
        statistic = log10(LR), extra has 'LR', 'per_locus_lr'.

    References
    ----------
    Gill, P., et al. (2006). DNA commission of the ISFG:
        Recommendations on the interpretation of mixtures.
        Forensic Sci. Int., 160(2-3), 90-101.
    """
    n_loci = len(mixture_alleles)
    if len(suspect_alleles) != n_loci or len(allele_freqs) != n_loci:
        raise ValueError("All inputs must have the same number of loci.")

    per_locus_lr = []

    for loc in range(n_loci):
        mix_set = set(mixture_alleles[loc])
        sus_geno = suspect_alleles[loc]
        freqs = allele_freqs[loc]

        sus_alleles_set = set(sus_geno)
        if not sus_alleles_set.issubset(mix_set):
            per_locus_lr.append(0.0)
            continue

        remaining = mix_set - sus_alleles_set
        all_alleles = list(freqs.keys())

        def _geno_prob(a1, a2):
            f1 = freqs.get(a1, 1e-6)
            f2 = freqs.get(a2, 1e-6)
            if theta > 0:
                if a1 == a2:
                    return (
                        (2 * theta + (1 - theta) * f1)
                        * (3 * theta + (1 - theta) * f1)
                        / ((1 + theta) * (1 + 2 * theta))
                    )
                else:
                    return (
                        2 * (theta + (1 - theta) * f1)
                        * (theta + (1 - theta) * f2)
                        / ((1 + theta) * (1 + 2 * theta))
                    )
            else:
                if a1 == a2:
                    return f1 ** 2
                return 2 * f1 * f2

        n_unknown = n_contributors - 1
        p_hp = 0.0
        p_hd = 0.0

        unknown_genos = list(_product(all_alleles, repeat=2))

        for combo in _product(unknown_genos, repeat=n_unknown):
            combo_alleles = set()
            prob = 1.0
            for a1, a2 in combo:
                combo_alleles.update([a1, a2])
                prob *= _geno_prob(a1, a2)

            full_hp = sus_alleles_set | combo_alleles
            if mix_set.issubset(full_hp):
                p_hp += prob

            full_hd_alleles = combo_alleles.copy()
            for a1_d, a2_d in _product(all_alleles, repeat=2):
                extra = {a1_d, a2_d}
                if mix_set.issubset(full_hd_alleles | extra):
                    p_hd += prob * _geno_prob(a1_d, a2_d)
                    break

        lr_loc = p_hp / max(p_hd, 1e-30) if p_hp > 0 else 1.0
        per_locus_lr.append(lr_loc)

    combined_lr = float(np.prod([x for x in per_locus_lr if x > 0]) if per_locus_lr else 1.0)
    log10_lr = np.log10(max(combined_lr, 1e-300))

    return GenomicsResult(
        name="MixtureLR",
        statistic=float(log10_lr),
        n=n_loci,
        extra={
            "LR": combined_lr,
            "per_locus_lr": per_locus_lr,
            "n_contributors": n_contributors,
        },
    )


def cheatsheet() -> str:
    return "dmxlr(mix, suspect, freqs) -> DNA mixture likelihood ratio."
