# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Catalan number computation."""

from math import comb

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The only true wisdom is in knowing you know nothing. — Socrates"


def catalan_number(n: int, **kwargs) -> DescriptiveResult:
    """
    Compute the n-th Catalan number.

    .. math::

        C_n = \\frac{1}{n+1}\\binom{2n}{n} = \\frac{(2n)!}{(n+1)!\\,n!}

    Catalan numbers count the number of distinct binary trees with *n*
    nodes, valid parenthesizations, Dyck paths, non-crossing partitions,
    and many other combinatorial structures.

    Asymptotically :math:`C_n \\sim 4^n / (n^{3/2} \\sqrt{\\pi})`.

    :param n: Non-negative integer index.
    :return: DescriptiveResult with C_n as value.
    :raises ValueError: If n < 0.

    References
    ----------
    Stanley, R. P. (2015). *Catalan Numbers*. Cambridge University Press.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}.")

    c_n = comb(2 * n, n) // (n + 1)

    return DescriptiveResult(
        name="catalan_number",
        value=float(c_n),
        extra={
            "C_n": c_n,
            "n": n,
            "asymptotic": 4.0**n / ((n + 1) ** 1.5 * np.sqrt(np.pi)) if n > 0 else 1.0,
        },
    )


catln = catalan_number


def cheatsheet() -> str:
    return "catalan_number({}) -> Catalan number computation."
