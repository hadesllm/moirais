# moirais.fn — function file (hadesllm/moirais)
"""Find irreducible polynomials over GF(2)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def irreducible_poly(m: int = 8) -> DescriptiveResult:
    """Find an irreducible polynomial of degree m over GF(2).

    :param m: Polynomial degree (2 <= m <= 13 for lookup, higher by search).
    :return: DescriptiveResult with polynomial as integer and binary string.
    """
    from moirais.crypto._gf2m import find_irreducible

    poly = find_irreducible(m)
    return DescriptiveResult(
        name="irreducible_poly",
        value=float(poly),
        extra={
            "polynomial": poly,
            "degree": m,
            "binary": bin(poly),
            "hex": hex(poly),
        },
    )


irpol = irreducible_poly


def cheatsheet() -> str:
    return "irreducible_poly({}) -> Find irreducible polynomials over GF(2)."
