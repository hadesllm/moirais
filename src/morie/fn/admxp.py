# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Admixture proportions estimation (maximum likelihood)."""

__all__ = ["admxp"]

import numpy as np

from ._containers import GenomicsResult


def admxp(
    Z: np.ndarray,
    *,
    K: int = 2,
    n_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> GenomicsResult:
    """Estimate individual admixture proportions via EM algorithm.

    Implements a simplified ADMIXTURE-like model that estimates
    individual ancestry proportions Q and ancestral allele frequencies
    P using an EM algorithm on genotype likelihoods.

    Parameters
    ----------
    Z : array, shape (n, p)
        Marker genotype matrix (0/1/2).
    K : int
        Number of ancestral populations.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    seed : int
        Random seed.

    Returns
    -------
    GenomicsResult
        statistic = final log-likelihood,
        extra has 'Q' (n x K admixture proportions),
        'P' (K x p ancestral allele frequencies).

    References
    ----------
    Alexander, D. H., Novembre, J., & Lange, K. (2009).
        Fast model-based estimation of ancestry in unrelated
        individuals. Genome Research, 19(9), 1655-1664.
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D.")
    n, p = Z.shape
    if K < 1 or n < K:
        raise ValueError(f"K must be in [1, {n}].")

    Q = rng.dirichlet(np.ones(K), size=n)
    P = np.clip(rng.uniform(0.1, 0.9, size=(K, p)), 0.01, 0.99)

    prev_ll = -np.inf

    for _ in range(n_iter):
        f = Q @ P
        f = np.clip(f, 1e-10, 1.0 - 1e-10)

        ll = np.sum(
            Z * np.log(f) + (2.0 - Z) * np.log(1.0 - f)
        )

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        w = np.zeros((n, K, p))
        for k in range(K):
            num = Q[:, k:k+1] * P[k:k+1, :]
            denom = f
            w[:, k, :] = num / np.maximum(denom, 1e-10)

        for k in range(K):
            P[k, :] = np.sum(w[:, k, :] * Z, axis=0) / np.maximum(
                np.sum(w[:, k, :] * 2.0, axis=0), 1e-10
            )
        P = np.clip(P, 0.01, 0.99)

        for i in range(n):
            Q[i, :] = np.sum(w[i, :, :] * Z[i, :], axis=1) + np.sum(
                w[i, :, :] * (2.0 - Z[i, :]), axis=1
            )
            Q[i, :] = np.maximum(Q[i, :], 1e-10)
            Q[i, :] /= np.sum(Q[i, :])

    return GenomicsResult(
        name="Admixture",
        statistic=float(prev_ll),
        n=n,
        extra={
            "Q": Q.tolist(),
            "P": P.tolist(),
            "K": K,
            "n_markers": p,
        },
    )


def cheatsheet() -> str:
    return "admxp(Z, K=2) -> Admixture proportions estimation (EM)."
