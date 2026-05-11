"""Spatial permutation test."""

from __future__ import annotations

from ._containers import DescriptiveResult


def permutation_test_spatial(Z, coords, stat_fn, n_perm=999):
    """Permutation test for a spatial statistic.

    Randomly reassigns observed values to locations and computes the
    distribution of the test statistic.

    .. epigraph:: "War. War never changes." -- Narrator, Fallout

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    stat_fn : callable
        Function ``stat_fn(Z, coords) -> float``.
    n_perm : int
        Number of permutations.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)

    observed = float(stat_fn(Z, coords))

    rng = np.random.default_rng(42)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        Z_perm = rng.permutation(Z)
        perm_stats[i] = stat_fn(Z_perm, coords)

    p_value = float(np.mean(perm_stats >= observed))

    return DescriptiveResult(
        name="permutation_test_spatial",
        value=observed,
        extra={
            "observed": observed,
            "p_value": p_value,
            "perm_mean": float(np.mean(perm_stats)),
            "perm_std": float(np.std(perm_stats)),
            "n_perm": n_perm,
        },
    )


sgpmt = permutation_test_spatial


def cheatsheet() -> str:
    return "permutation_test_spatial({}) -> Spatial permutation test."
