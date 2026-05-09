# moirais.fn — function file (hadesllm/moirais)
"""Relapse vs reinfection probability for P. vivax (Pv3Rs)."""

__all__ = ["relps"]

import numpy as np

from ._containers import GenomicsResult


def relps(
    alleles_initial: np.ndarray,
    alleles_recurrent: np.ndarray,
    *,
    allele_freqs: np.ndarray | None = None,
    n_loci: int | None = None,
) -> GenomicsResult:
    """Classify recurrent P. vivax episode as relapse vs reinfection.

    Uses multilocus microsatellite or SNP genotyping data to compute
    the probability that a recurrent episode is a relapse (same
    parasite clone) versus a new infection, based on identity at
    typed loci and population allele frequencies.

    Parameters
    ----------
    alleles_initial : array, shape (L,) or (L, max_alleles)
        Allele calls from the initial episode at L loci.
        If 1-D, treated as single allele per locus.
    alleles_recurrent : array, shape (L,) or (L, max_alleles)
        Allele calls from the recurrent episode.
    allele_freqs : array, shape (L,), optional
        Population frequency of the observed allele at each locus.
        Required if computing exact probabilities.  If None, uniform
        frequency 1/n_alleles is assumed.
    n_loci : int, optional
        Override number of loci (for partial data).

    Returns
    -------
    GenomicsResult
        statistic = P(relapse | data), p_value = P(reinfection),
        extra has 'matching_loci', 'n_loci', 'log_lr'.

    References
    ----------
    Taylor, A. R., et al. (2019). Quantifying connectivity between
        local Plasmodium falciparum malaria parasite populations
        using identity by descent. PLoS Genetics, 15(10), e1008384.
    Commons, R. J., et al. (2020). Estimating the proportion of
        Plasmodium vivax recurrences caused by relapse. PLoS Med.
    """
    a0 = np.asarray(alleles_initial)
    a1 = np.asarray(alleles_recurrent)

    if a0.ndim == 1:
        a0 = a0.reshape(-1, 1)
    if a1.ndim == 1:
        a1 = a1.reshape(-1, 1)

    L = n_loci if n_loci is not None else a0.shape[0]
    if a0.shape[0] != a1.shape[0]:
        raise ValueError("Initial and recurrent must have same number of loci.")

    if allele_freqs is not None:
        pf = np.asarray(allele_freqs, dtype=float).ravel()
        if len(pf) != L:
            raise ValueError("allele_freqs length must match number of loci.")
    else:
        pf = np.full(L, 0.1)

    pf = np.clip(pf, 1e-6, 1.0 - 1e-6)

    matches = 0
    log_lr = 0.0

    for locus in range(L):
        a0_set = set(a0[locus].ravel())
        a1_set = set(a1[locus].ravel())
        shared = a0_set & a1_set
        if len(shared) > 0:
            matches += 1
            log_lr += np.log(1.0 / max(pf[locus], 1e-10))
        else:
            log_lr += np.log(pf[locus])

    lr = np.exp(np.clip(log_lr, -500, 500))
    p_relapse = lr / (1.0 + lr)
    p_reinfection = 1.0 - p_relapse

    return GenomicsResult(
        name="Pv3Rs_relapse",
        statistic=float(p_relapse),
        p_value=float(p_reinfection),
        n=L,
        extra={
            "matching_loci": matches,
            "n_loci": L,
            "log_lr": float(log_lr),
        },
    )


def cheatsheet() -> str:
    return "relps(a0, a1) -> P. vivax relapse vs reinfection probability."
