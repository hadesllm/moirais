# moirais.fn — function file (hadesllm/moirais)
"""Genomic relationship matrix (VanRaden methods 1 and 2)."""

__all__ = ["gmatx"]

import numpy as np

from ._containers import GenomicsResult


def gmatx(
    Z: np.ndarray,
    *,
    method: int = 1,
    allele_freqs: np.ndarray | None = None,
) -> GenomicsResult:
    """Compute the genomic relationship matrix (GRM).

    Parameters
    ----------
    Z : array, shape (n, p)
        Marker genotype matrix coded as 0, 1, 2 (minor allele counts).
    method : {1, 2}
        VanRaden method.

        * Method 1: G = M M' / (2 sum p_j (1 - p_j))
        * Method 2: G = (1/p) sum_j m_j m_j' / (2 p_j (1 - p_j))

        where M is the centered genotype matrix and p_j are allele
        frequencies.
    allele_freqs : array, shape (p,), optional
        Known allele frequencies.  If None, estimated from Z.

    Returns
    -------
    GenomicsResult
        statistic = mean diagonal of G,
        extra has 'G' (the n x n GRM as nested list).

    References
    ----------
    VanRaden, P. M. (2008). Efficient methods to compute genomic
        predictions. J. Dairy Sci., 91(11), 4414-4423.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D (n x p).")
    n, p = Z.shape

    if allele_freqs is not None:
        pf = np.asarray(allele_freqs, dtype=float).ravel()
        if len(pf) != p:
            raise ValueError("allele_freqs length must equal number of markers.")
    else:
        pf = np.mean(Z, axis=0) / 2.0

    pf = np.clip(pf, 1e-6, 1.0 - 1e-6)
    M = Z - 2.0 * pf[np.newaxis, :]

    if method == 1:
        denom = 2.0 * np.sum(pf * (1.0 - pf))
        if denom < 1e-12:
            denom = 1.0
        G = (M @ M.T) / denom
    elif method == 2:
        scaling = 2.0 * pf * (1.0 - pf)
        scaling = np.where(scaling < 1e-12, 1.0, scaling)
        M_scaled = M / np.sqrt(scaling)[np.newaxis, :]
        G = (M_scaled @ M_scaled.T) / p
    else:
        raise ValueError("method must be 1 or 2.")

    return GenomicsResult(
        name="GRM",
        statistic=float(np.mean(np.diag(G))),
        n=n,
        extra={"G": G.tolist(), "method": method, "n_markers": p},
    )


def cheatsheet() -> str:
    return "gmatx(Z) -> Genomic relationship matrix (VanRaden method 1 or 2)."
