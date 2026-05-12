# morie.fn — function file (hadesllm/morie)
"""Permutation count P(n, k)."""

from math import perm as _perm

from ._containers import DescriptiveResult

_QUOTE = "Look well into thyself; there is a source which will always spring up. — Marcus Aurelius"


def permutation_count(n: int, k: int, **kwargs) -> DescriptiveResult:
    r"""
    Compute the number of k-permutations of n items.

    .. math::

        P(n, k) = \\frac{n!}{(n-k)!} = n \\cdot (n-1) \\cdots (n-k+1)

    Uses Python's exact-integer ``math.perm`` for arbitrary precision.

    :param n: Total items (n >= 0).
    :param k: Items to arrange (0 <= k <= n).
    :return: DescriptiveResult with P(n,k) as value.
    :raises ValueError: If k < 0 or k > n or n < 0.

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

    result = _perm(n, k)

    from scipy.special import gammaln

    log_perm = gammaln(n + 1) - gammaln(n - k + 1)

    return DescriptiveResult(
        name="permutation_count",
        value=float(result),
        extra={
            "P_n_k": result,
            "log_P_n_k": float(log_perm),
            "n": n,
            "k": k,
        },
    )


permu = permutation_count


def cheatsheet() -> str:
    return "permutation_count({}) -> Permutation count P(n, k)."
