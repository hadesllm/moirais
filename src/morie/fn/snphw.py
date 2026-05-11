"""SNP Hardy-Weinberg exact test."""

__all__ = ["snphw"]

import numpy as np

from ._containers import GenomicsResult


def snphw(
    genotype_counts: np.ndarray | list,
) -> GenomicsResult:
    """Hardy-Weinberg exact test for a biallelic SNP.

    Uses the exact probability method (complete enumeration) for
    genotype counts (AA, AB, BB).

    Parameters
    ----------
    genotype_counts : array-like, shape (3,) or (m, 3)
        Counts of [n_AA, n_AB, n_BB] for each SNP.
        If 2-D, each row is a separate SNP.

    Returns
    -------
    GenomicsResult
        statistic = chi-squared test statistic (for single SNP),
        p_value = exact p-value,
        extra has 'p_values' for multi-SNP input.

    References
    ----------
    Wigginton, J. E., Cutler, D. J., & Abecasis, G. R. (2005).
        A note on exact tests of Hardy-Weinberg equilibrium. Am. J.
        Hum. Genet., 76(5), 887-893.
    """
    counts = np.asarray(genotype_counts, dtype=int)
    if counts.ndim == 1:
        counts = counts.reshape(1, -1)
    if counts.shape[1] != 3:
        raise ValueError("Each row must have 3 genotype counts (AA, AB, BB).")
    if np.any(counts < 0):
        raise ValueError("Counts must be non-negative.")

    p_values = []
    chi2_vals = []

    for row in counts:
        n_aa, n_ab, n_bb = int(row[0]), int(row[1]), int(row[2])
        n = n_aa + n_ab + n_bb
        if n == 0:
            p_values.append(1.0)
            chi2_vals.append(0.0)
            continue

        n_a = 2 * n_aa + n_ab
        n_b = 2 * n_bb + n_ab
        p_a = n_a / (2.0 * n)

        exp_aa = p_a ** 2 * n
        exp_ab = 2.0 * p_a * (1.0 - p_a) * n
        exp_bb = (1.0 - p_a) ** 2 * n

        chi2 = 0.0
        for obs, exp in [(n_aa, exp_aa), (n_ab, exp_ab), (n_bb, exp_bb)]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
        chi2_vals.append(float(chi2))

        if n_a == 0 or n_b == 0:
            p_values.append(1.0)
            continue

        min_het = n_a % 2
        max_het = min(n_a, n_b)

        def _het_prob(het):
            aa = (n_a - het) // 2
            bb = (n_b - het) // 2
            log_p = (
                np.sum(np.log(np.arange(1, n + 1)))
                - np.sum(np.log(np.arange(1, aa + 1)))
                - np.sum(np.log(np.arange(1, het + 1)))
                - np.sum(np.log(np.arange(1, bb + 1)))
                + het * np.log(2)
                + np.sum(np.log(np.arange(1, n_a + 1)))
                + np.sum(np.log(np.arange(1, n_b + 1)))
                - np.sum(np.log(np.arange(1, 2 * n + 1)))
            )
            return np.exp(log_p)

        probs = {}
        for het in range(min_het, max_het + 1, 2):
            probs[het] = _het_prob(het)

        total = sum(probs.values())
        if total > 0:
            for k in probs:
                probs[k] /= total

        obs_prob = probs.get(n_ab, 0.0)
        p_val = sum(p for p in probs.values() if p <= obs_prob + 1e-12)
        p_values.append(float(min(p_val, 1.0)))

    if len(p_values) == 1:
        return GenomicsResult(
            name="HWE_exact",
            statistic=chi2_vals[0],
            p_value=p_values[0],
            n=int(np.sum(counts[0])),
        )

    return GenomicsResult(
        name="HWE_exact",
        statistic=float(np.mean(chi2_vals)),
        p_value=float(np.median(p_values)),
        n=int(np.sum(counts[0])),
        extra={"p_values": p_values, "chi2_values": chi2_vals},
    )


def cheatsheet() -> str:
    return "snphw(genotype_counts) -> SNP Hardy-Weinberg exact test."
