# moirais.fn — function file (hadesllm/moirais)
"""Recrudescence probability estimation."""

__all__ = ["recrd"]

import numpy as np

from ._containers import GenomicsResult


def recrd(
    alleles_pre: np.ndarray,
    alleles_post: np.ndarray,
    *,
    allele_freqs: np.ndarray | None = None,
    prior_recrud: float = 0.5,
) -> GenomicsResult:
    """Estimate probability of recrudescence vs new infection.

    Compares parasite genotypes before treatment and at recurrence
    to distinguish treatment failure (recrudescence) from new
    infection using a Bayesian framework.

    Parameters
    ----------
    alleles_pre : array, shape (L,)
        Allele calls at L loci before treatment.
    alleles_post : array, shape (L,)
        Allele calls at L loci at recurrence.
    allele_freqs : array, shape (L,), optional
        Population frequency of the matching allele at each locus.
        If None, assumes 0.1 per locus.
    prior_recrud : float
        Prior probability of recrudescence (default 0.5).

    Returns
    -------
    GenomicsResult
        statistic = P(recrudescence | data),
        p_value = P(new infection | data).

    References
    ----------
    WHO (2008). Methods and techniques for clinical trials on
        antimalarial drug efficacy: genotyping to identify parasite
        populations.
    """
    a_pre = np.asarray(alleles_pre).ravel()
    a_post = np.asarray(alleles_post).ravel()
    L = len(a_pre)

    if len(a_post) != L:
        raise ValueError("Pre and post must have same number of loci.")

    if allele_freqs is not None:
        pf = np.asarray(allele_freqs, dtype=float).ravel()
    else:
        pf = np.full(L, 0.1)

    pf = np.clip(pf, 1e-6, 1.0 - 1e-6)

    log_lr = 0.0
    matches = 0
    for i in range(L):
        if a_pre[i] == a_post[i]:
            matches += 1
            log_lr += np.log(1.0) - np.log(pf[i])
        else:
            log_lr += np.log(1e-4) - np.log(1.0 - pf[i])

    log_prior_odds = np.log(max(prior_recrud, 1e-10)) - np.log(max(1 - prior_recrud, 1e-10))
    log_post_odds = log_prior_odds + log_lr
    post_odds = np.exp(np.clip(log_post_odds, -500, 500))
    p_recrud = post_odds / (1.0 + post_odds)

    return GenomicsResult(
        name="Recrudescence",
        statistic=float(p_recrud),
        p_value=float(1.0 - p_recrud),
        n=L,
        extra={
            "matching_loci": matches,
            "n_loci": L,
            "log_lr": float(log_lr),
            "prior": prior_recrud,
        },
    )


def cheatsheet() -> str:
    return "recrd(pre, post) -> Recrudescence probability estimation."
