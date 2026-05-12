# morie.fn -- function file (hadesllm/morie)
"""Knox test for space-time clustering."""

import numpy as np

from ._containers import TestResult


def knox_test(
    locations: np.ndarray,
    times: np.ndarray,
    s_threshold: float,
    t_threshold: float,
    n_permutations: int = 999,
    seed: int = 42,
) -> TestResult:
    """
    Knox test for space-time interaction in point events.

    Counts pairs close in both space and time and compares to
    a permutation null distribution.

    :param locations: (n, 2) spatial coordinates.
    :param times: (n,) event times.
    :param s_threshold: Spatial proximity threshold.
    :param t_threshold: Temporal proximity threshold.
    :param n_permutations: Number of permutations for p-value.
    :param seed: Random seed.
    :return: TestResult with Knox statistic and permutation p-value.

    References
    ----------
    Knox EG (1964). The detection of space-time interactions.
    Applied Statistics, 13(1), 25-30.
    """
    locs = np.asarray(locations, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).ravel()
    n = locs.shape[0]
    dmat = np.sqrt(((locs[:, None, :] - locs[None, :, :]) ** 2).sum(axis=2))
    tmat = np.abs(t[:, None] - t[None, :])
    np.fill_diagonal(dmat, np.inf)
    np.fill_diagonal(tmat, np.inf)
    close_s = dmat <= s_threshold
    close_t = tmat <= t_threshold
    observed = int(np.sum(close_s & close_t)) // 2
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_permutations):
        t_perm = rng.permutation(t)
        tmat_p = np.abs(t_perm[:, None] - t_perm[None, :])
        np.fill_diagonal(tmat_p, np.inf)
        perm_stat = int(np.sum(close_s & (tmat_p <= t_threshold))) // 2
        if perm_stat >= observed:
            count_ge += 1
    pval = (count_ge + 1) / (n_permutations + 1)
    return TestResult(
        test_name="knox_test",
        statistic=float(observed),
        p_value=float(pval),
        method="permutation",
        n=n,
        extra={"s_threshold": s_threshold, "t_threshold": t_threshold, "n_permutations": n_permutations},
    )


knox = knox_test


def cheatsheet() -> str:
    return "knox_test({}) -> Knox test for space-time clustering."
