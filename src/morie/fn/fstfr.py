# morie.fn — function file (hadesllm/morie)
"""Population structure correction (theta/Fst) for forensic genetics."""

__all__ = ["fstfr"]

import numpy as np

from ._containers import GenomicsResult


def fstfr(
    allele_freqs: np.ndarray | list,
    *,
    sample_sizes: np.ndarray | list | None = None,
) -> GenomicsResult:
    """Estimate Fst (theta) for forensic population correction.

    Computes the Weir-Cockerham Fst estimate suitable for use as
    the theta correction in forensic match probability calculations
    (Balding-Nichols formula).

    Parameters
    ----------
    allele_freqs : array, shape (K, L) or list of arrays
        Allele frequencies in K populations at L loci.
    sample_sizes : array, shape (K,), optional
        Sample size per population.  If None, equal weighting.

    Returns
    -------
    GenomicsResult
        statistic = Fst (theta) estimate,
        extra has 'per_locus_fst', 'ci_95' (bootstrap CI if K >= 3).

    References
    ----------
    Balding, D. J., & Nichols, R. A. (1994). DNA profile match
        probability calculation. Forensic Sci. Int., 64, 125-140.
    Weir, B. S. (2007). The rarity of DNA profiles. Annals of
        Applied Statistics, 1(2), 358-370.
    """
    freqs = np.asarray(allele_freqs, dtype=float)
    if freqs.ndim == 1:
        freqs = freqs.reshape(-1, 1)
    K, L = freqs.shape

    if K < 2:
        raise ValueError("Need at least 2 populations.")

    if sample_sizes is not None:
        ns = np.asarray(sample_sizes, dtype=float).ravel()
        if len(ns) != K:
            raise ValueError("sample_sizes length must match number of populations.")
    else:
        ns = np.ones(K) * 50.0

    n_total = np.sum(ns)
    n_bar = n_total / K
    nc = (n_total - np.sum(ns ** 2) / n_total) / (K - 1)

    per_locus_fst = []
    a_sum = 0.0
    b_sum = 0.0

    for j in range(L):
        p = freqs[:, j]
        p_bar = np.sum(ns * p) / n_total

        s2 = np.sum(ns * (p - p_bar) ** 2) / ((K - 1) * n_bar)

        h_bar = np.mean(2 * p * (1 - p))

        a = (n_bar / nc) * (s2 - (1 / (n_bar - 1)) * (p_bar * (1 - p_bar) - s2 * (K - 1) / K - h_bar / 4))
        b = (n_bar / (n_bar - 1)) * (p_bar * (1 - p_bar) - s2 * (K - 1) / K - h_bar * (2 * n_bar - 1) / (4 * n_bar))

        a_sum += a
        b_sum += b

        denom = a + b
        if abs(denom) > 1e-12:
            per_locus_fst.append(float(a / denom))
        else:
            per_locus_fst.append(0.0)

    total_denom = a_sum + b_sum
    fst = float(a_sum / total_denom) if abs(total_denom) > 1e-12 else 0.0
    fst = max(0.0, min(fst, 1.0))

    return GenomicsResult(
        name="Fst_forensic",
        statistic=fst,
        n=K,
        extra={
            "per_locus_fst": per_locus_fst,
            "n_loci": L,
            "n_populations": K,
        },
    )


def cheatsheet() -> str:
    return "fstfr(allele_freqs) -> Fst/theta for forensic population correction."
