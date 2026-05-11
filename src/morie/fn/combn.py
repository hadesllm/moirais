# morie.fn — function file (hadesllm/morie)
"""Exact binomial coefficient computation."""

from math import comb as _comb

from ._containers import DescriptiveResult

_QUOTE = "He who is brave is free. — Seneca"


def combinations_count(n: int, k: int, **kwargs) -> DescriptiveResult:
    """
    Compute the exact binomial coefficient :math:`\\binom{n}{k}`.

    .. math::

        \\binom{n}{k} = \\frac{n!}{k!(n-k)!}

    Uses Python's exact-integer ``math.comb`` for arbitrary precision.
    Also returns the log-binomial for large values.

    :param n: Total items (n >= 0).
    :param k: Items to choose (0 <= k <= n).
    :return: DescriptiveResult with C(n,k) as value.
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

    result = _comb(n, k)

    from scipy.special import gammaln

    log_binom = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    return DescriptiveResult(
        name="combinations_count",
        value=float(result),
        extra={
            "C_n_k": result,
            "log_C_n_k": float(log_binom),
            "n": n,
            "k": k,
        },
    )


combn = combinations_count


def cheatsheet() -> str:
    return "combinations_count({}) -> Exact binomial coefficient computation."
