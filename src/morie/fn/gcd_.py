# morie.fn -- function file (hadesllm/morie)
"""Extended Euclidean algorithm."""


from ._containers import DescriptiveResult
def extended_gcd(a: int, b: int, **kwargs) -> DescriptiveResult:
    r"""
    Extended Euclidean algorithm returning gcd, x, y such that ax + by = gcd.

    .. math::

        \\gcd(a, b) = a \\cdot x + b \\cdot y

    Also known as Bezout's identity. The algorithm runs in
    :math:`O(\\log(\\min(a, b)))` time.

    :param a: First integer.
    :param b: Second integer.
    :return: DescriptiveResult with gcd as value and Bezout coefficients.

    References
    ----------
    Knuth, D. E. (1997). *The Art of Computer Programming*, Vol. 2:
    Seminumerical Algorithms (3rd ed.). Addison-Wesley.
    """
    a, b = int(a), int(b)

    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    if a < 0:
        old_s = -old_s
    if b < 0:
        old_t = -old_t

    return DescriptiveResult(
        name="extended_gcd",
        value=float(old_r),
        extra={
            "gcd": old_r,
            "x": old_s,
            "y": old_t,
            "a": a,
            "b": b,
            "verification": a * old_s + b * old_t,
        },
    )


gcd_ = extended_gcd


def cheatsheet() -> str:
    return "extended_gcd({}) -> Extended Euclidean algorithm."
