# morie.fn -- function file (hadesllm/morie)
"""Gap statistic for optimal k. 'The greatest teacher, failure is.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gap_statistic(
    X: np.ndarray,
    max_k: int = 10,
    n_refs: int = 10,
    seed: int = 42,
) -> DescriptiveResult:
    """
    Gap statistic for choosing optimal number of clusters.

    Compares within-cluster dispersion to that expected under a null
    reference distribution (uniform over the data range).

    :param X: Data matrix (n_samples, n_features).
    :param max_k: Maximum k to evaluate. Default 10.
    :param n_refs: Number of reference datasets. Default 10.
    :param seed: RNG seed. Default 42.
    :return: DescriptiveResult with optimal k as value.
    :raises ValueError: If X not 2-D or max_k < 2.

    References
    ----------
    Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number
    of clusters in a data set via the gap statistic. Journal of the Royal
    Statistical Society B, 63(2), 411--423. doi:10.1111/1467-9868.00293
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("X must be a 2-D array with >= 3 samples.")
    if max_k < 2:
        raise ValueError(f"max_k must be >= 2, got {max_k}.")

    rng = np.random.default_rng(seed)
    n, p = X.shape
    max_k = min(max_k, n - 1)

    def _kmeans_wk(data: np.ndarray, k: int) -> float:
        from scipy.cluster.vq import kmeans2

        centroids, lab = kmeans2(data, k, minit="points", seed=rng.integers(2**31))
        wk = 0.0
        for c in range(k):
            mask = lab == c
            if np.sum(mask) > 0:
                wk += np.sum((data[mask] - centroids[c]) ** 2)
        return wk

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    gaps = np.zeros(max_k)
    sks = np.zeros(max_k)
    log_wks = np.zeros(max_k)

    for k in range(1, max_k + 1):
        wk = _kmeans_wk(X, k)
        log_wk = np.log(wk + 1e-300)
        log_wks[k - 1] = log_wk

        ref_log_wks = np.zeros(n_refs)
        for r in range(n_refs):
            ref = rng.uniform(mins, maxs, size=(n, p))
            ref_wk = _kmeans_wk(ref, k)
            ref_log_wks[r] = np.log(ref_wk + 1e-300)

        gaps[k - 1] = np.mean(ref_log_wks) - log_wk
        sks[k - 1] = np.std(ref_log_wks, ddof=0) * np.sqrt(1 + 1.0 / n_refs)

    optimal_k = 1
    for k in range(max_k - 1):
        if gaps[k] >= gaps[k + 1] - sks[k + 1]:
            optimal_k = k + 1
            break
    else:
        optimal_k = int(np.argmax(gaps)) + 1

    return DescriptiveResult(
        name="Gap Statistic",
        value=optimal_k,
        extra={
            "optimal_k": optimal_k,
            "gaps": gaps.tolist(),
            "sks": sks.tolist(),
            "log_wks": log_wks.tolist(),
            "max_k": max_k,
            "n_refs": n_refs,
        },
    )


gapst = gap_statistic


def cheatsheet() -> str:
    return "gap_statistic({}) -> Gap statistic for optimal k. 'The greatest teacher, failure "
