# moirais.fn — function file (hadesllm/moirais)
"""Modular multiplicative inverse."""


from ._containers import DescriptiveResult

_QUOTE = "You have power over your mind — not outside events. — Marcus Aurelius"


def mod_inverse(a: int, m: int, **kwargs) -> DescriptiveResult:
    """
    Compute the modular multiplicative inverse of *a* modulo *m*.

    Finds :math:`x` such that :math:`a \\cdot x \\equiv 1 \\pmod{m}`,
    which exists iff :math:`\\gcd(a, m) = 1`.

    Uses the extended Euclidean algorithm in :math:`O(\\log m)` time.

    :param a: Integer whose inverse is sought.
    :param m: Modulus (must be > 1).
    :return: DescriptiveResult with inverse as value.
    :raises ValueError: If m <= 1 or inverse does not exist.

    References
    ----------
    Knuth, D. E. (1997). *The Art of Computer Programming*, Vol. 2:
    Seminumerical Algorithms (3rd ed.). Addison-Wesley.
    """
    a, m = int(a), int(m)
    if m <= 1:
        raise ValueError(f"Modulus must be > 1, got {m}.")

    def _ext_gcd(a_, b_):
        if a_ == 0:
            return b_, 0, 1
        g, x, y = _ext_gcd(b_ % a_, a_)
        return g, y - (b_ // a_) * x, x

    g, x, _ = _ext_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Inverse of {a} mod {m} does not exist (gcd={g}).")

    inv = x % m

    return DescriptiveResult(
        name="mod_inverse",
        value=float(inv),
        extra={
            "inverse": inv,
            "a": a,
            "m": m,
            "verification": (a * inv) % m,
        },
    )


modin = mod_inverse


def cheatsheet() -> str:
    return "mod_inverse({}) -> Modular multiplicative inverse."
