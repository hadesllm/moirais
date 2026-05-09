"""Stirling numbers of the first and second kind."""

import numpy as np

from ._containers import DescriptiveResult
def stirling_number(n: int, k: int, kind: int = 2, **kwargs) -> DescriptiveResult:
    """
    Compute the Stirling number S(n, k) of the first or second kind.

    **Second kind** :math:`S(n, k)` = number of ways to partition a set
    of *n* elements into *k* non-empty subsets:

    .. math::

        S(n, k) = k \\cdot S(n-1, k) + S(n-1, k-1)

    **First kind** (unsigned) :math:`|s(n, k)|` = number of permutations
    of *n* with exactly *k* cycles:

    .. math::

        |s(n, k)| = (n-1) \\cdot |s(n-1, k)| + |s(n-1, k-1)|

    :param n: Total elements (n >= 0).
    :param k: Number of subsets/cycles (0 <= k <= n).
    :param kind: 1 for first kind (unsigned), 2 for second kind. Default 2.
    :return: DescriptiveResult with Stirling number as value.
    :raises ValueError: If k < 0 or k > n or kind not in {1, 2}.

    References
    ----------
    Graham, R. L., Knuth, D. E. & Patashnik, O. (1994). *Concrete
    Mathematics* (2nd ed.). Addison-Wesley.
    """
    n, k = int(n), int(k)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}.")
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, {n}], got {k}.")
    if kind not in (1, 2):
        raise ValueError(f"kind must be 1 or 2, got {kind}.")

    dp = np.zeros((n + 1, k + 1), dtype=np.int64)
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            if kind == 2:
                dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1]
            else:
                dp[i][j] = (i - 1) * dp[i - 1][j] + dp[i - 1][j - 1]

    result = int(dp[n][k])

    return DescriptiveResult(
        name="stirling_number",
        value=float(result),
        extra={
            "S_n_k": result,
            "n": n,
            "k": k,
            "kind": kind,
            "kind_name": "first" if kind == 1 else "second",
        },
    )


stirn = stirling_number


def cheatsheet() -> str:
    return "stirling_number({}) -> Stirling numbers of the first and second kind."
