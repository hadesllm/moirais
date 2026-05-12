# morie.fn -- function file (hadesllm/morie)
"""Dedekind sum s(h, k)."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def dedekind_sum(
    h: int = 1,
    k: int = 7,
) -> DescriptiveResult:
    r"""Compute the Dedekind sum s(h, k).

    .. math::

        s(h, k) = \\sum_{r=1}^{k-1}
                   \\left(\\left(\\frac{r}{k}\\right)\\right)
                   \\left(\\left(\\frac{hr}{k}\\right)\\right)

    where :math:`((x)) = x - \\lfloor x \\rfloor - 1/2` if x not integer, else 0.

    Dedekind sums appear in the transformation law of the Dedekind eta function,
    central to string partition functions.

    :param h: Integer, coprime to k.
    :param k: Positive integer.
    :return: DescriptiveResult with the Dedekind sum value.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")

    def sawtooth(x: float) -> float:
        if abs(x - round(x)) < 1e-15:
            return 0.0
        return x - math.floor(x) - 0.5

    s = 0.0
    for r in range(1, k):
        s += sawtooth(r / k) * sawtooth(h * r / k)
    return DescriptiveResult(
        name="dedekind_sum",
        value=s,
        extra={"h": h, "k": k, "s_hk": s, "gcd": math.gcd(abs(h), k)},
    )


def cheatsheet() -> str:
    return "dedekind_sum(h, k) -> Dedekind sum s(h,k)"


dednm = dedekind_sum
