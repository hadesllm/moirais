# moirais.fn — function file (hadesllm/moirais)
"""Genomic block bootstrap."""

__all__ = ["gnblk"]

import numpy as np

from ._containers import GenomicsResult


def gnblk(
    statistic_fn,
    Z: np.ndarray,
    *,
    block_size: int = 50,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> GenomicsResult:
    """Block bootstrap for genomic statistics.

    Accounts for linkage disequilibrium by resampling contiguous
    blocks of markers rather than individual SNPs.

    Parameters
    ----------
    statistic_fn : callable
        Function that takes a genotype matrix (n x p') and returns
        a scalar statistic.
    Z : array, shape (n, p)
        Marker genotype matrix (0/1/2), ordered by genomic position.
    block_size : int
        Number of contiguous markers per block.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int
        Random seed.

    Returns
    -------
    GenomicsResult
        statistic = original statistic value,
        extra has 'se' (bootstrap SE), 'ci_lower', 'ci_upper'
        (95% CI), 'bootstrap_distribution'.

    References
    ----------
    Kunsch, H. R. (1989). The jackknife and the bootstrap for
        general stationary observations. Annals of Statistics,
        17(3), 1217-1241.
    Fitzpatrick, B. M. (2012). Estimating ancestry and heterozygosity
        of hybrids using molecular markers. BMC Evol. Biol., 12, 131.
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape

    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if block_size > p:
        block_size = p

    n_blocks = max(p // block_size, 1)
    block_starts = np.arange(0, p, block_size)

    orig_stat = float(statistic_fn(Z))

    boot_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_blocks = rng.choice(len(block_starts), size=len(block_starts), replace=True)
        cols = []
        for bi in sampled_blocks:
            start = block_starts[bi]
            end = min(start + block_size, p)
            cols.append(Z[:, start:end])
        Z_boot = np.hstack(cols)
        boot_stats[b] = statistic_fn(Z_boot)

    se = float(np.std(boot_stats, ddof=1))
    ci_lower = float(np.percentile(boot_stats, 2.5))
    ci_upper = float(np.percentile(boot_stats, 97.5))
    bias = float(np.mean(boot_stats) - orig_stat)

    return GenomicsResult(
        name="GenomicBlockBootstrap",
        statistic=orig_stat,
        n=n,
        extra={
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "bias": bias,
            "n_bootstrap": n_bootstrap,
            "block_size": block_size,
            "n_blocks": len(block_starts),
        },
    )


def cheatsheet() -> str:
    return "gnblk(fn, Z) -> Genomic block bootstrap with LD correction."
