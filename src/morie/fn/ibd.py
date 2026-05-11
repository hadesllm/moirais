# morie.fn — function file (hadesllm/morie)
"""Identity by descent (IBD) estimation from SNP data."""

__all__ = ["ibd"]

import numpy as np

from ._containers import GenomicsResult


def ibd(
    Z: np.ndarray,
    *,
    allele_freqs: np.ndarray | None = None,
) -> GenomicsResult:
    """Estimate pairwise IBD sharing (method of moments).

    Computes pairwise IBD proportions (pi-hat) using the method of
    moments estimator based on IBS (identity by state) sharing and
    population allele frequencies.

    Parameters
    ----------
    Z : array, shape (n, p)
        Marker genotype matrix (0/1/2).
    allele_freqs : array, shape (p,), optional
        Allele frequencies.  If None, estimated from the data.

    Returns
    -------
    GenomicsResult
        statistic = mean pi-hat across all pairs,
        extra has 'pi_hat' (n x n matrix of pairwise IBD estimates),
        'Z0', 'Z1', 'Z2' (probabilities of sharing 0, 1, 2 alleles IBD).

    References
    ----------
    Purcell, S., et al. (2007). PLINK: a tool set for whole-genome
        association and population-based linkage analyses. Am. J.
        Hum. Genet., 81(3), 559-575.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D.")
    n, p = Z.shape
    if n < 2:
        raise ValueError("Need at least 2 individuals.")

    if allele_freqs is not None:
        pf = np.asarray(allele_freqs, dtype=float).ravel()
    else:
        pf = np.mean(Z, axis=0) / 2.0

    pf = np.clip(pf, 1e-6, 1.0 - 1e-6)

    pi_hat = np.zeros((n, n))
    z0_mat = np.zeros((n, n))
    z1_mat = np.zeros((n, n))
    z2_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ibs0 = 0
            ibs1 = 0
            ibs2 = 0
            e_ibs0 = 0.0
            e_ibs1 = 0.0

            for k in range(p):
                gi, gj = Z[i, k], Z[j, k]
                pk = pf[k]
                qk = 1.0 - pk

                diff = abs(gi - gj)
                if diff > 1.5:
                    ibs0 += 1
                elif diff > 0.5:
                    ibs1 += 1
                else:
                    ibs2 += 1

                e_ibs0 += (pk ** 2 * qk ** 2) * 2.0
                e_ibs1 += 2.0 * pk * qk * (pk ** 2 + qk ** 2)

            valid = p
            if valid > 0:
                f_ibs0 = ibs0 / valid
                f_ibs1 = ibs1 / valid
                e0 = e_ibs0 / valid
                e1 = e_ibs1 / valid

                if e0 > 1e-10:
                    z0 = f_ibs0 / e0
                else:
                    z0 = 0.0
                z0 = min(max(z0, 0.0), 1.0)

                z2 = max(1.0 - z0 - (f_ibs1 - e1 * z0) / max(0.5 - e1, 1e-10), 0.0)
                z2 = min(z2, 1.0)
                z1_val = max(1.0 - z0 - z2, 0.0)

                pi_hat_ij = z1_val * 0.5 + z2
            else:
                z0 = z1_val = z2 = 0.0
                pi_hat_ij = 0.0

            pi_hat[i, j] = pi_hat[j, i] = pi_hat_ij
            z0_mat[i, j] = z0_mat[j, i] = z0
            z1_mat[i, j] = z1_mat[j, i] = z1_val
            z2_mat[i, j] = z2_mat[j, i] = z2

    np.fill_diagonal(pi_hat, 1.0)
    np.fill_diagonal(z2_mat, 1.0)

    upper = pi_hat[np.triu_indices(n, k=1)]
    mean_pi = float(np.mean(upper)) if len(upper) > 0 else 0.0

    return GenomicsResult(
        name="IBD",
        statistic=mean_pi,
        n=n,
        extra={
            "pi_hat": pi_hat.tolist(),
            "Z0": z0_mat.tolist(),
            "Z1": z1_mat.tolist(),
            "Z2": z2_mat.tolist(),
            "n_markers": p,
        },
    )


def cheatsheet() -> str:
    return "ibd(Z) -> Pairwise identity-by-descent estimation."
