# morie.fn -- function file (hadesllm/morie)
"""GF(2^m) finite field arithmetic."""

from __future__ import annotations

from ._containers import DescriptiveResult


def gf2m_arithmetic(a: int, b: int, m: int = 8, op: str = "mul") -> DescriptiveResult:
    """Perform arithmetic in GF(2^m).

    :param a: First element (integer representation).
    :param b: Second element (integer representation).
    :param m: Field extension degree.
    :param op: Operation -- "add", "mul", "inv" (ignores b), "pow" (b is exponent).
    :return: DescriptiveResult with the computed value.
    """
    from morie.crypto._gf2m import find_irreducible, gf2m_add, gf2m_inv, gf2m_mul, gf2m_pow

    mod_poly = find_irreducible(m)

    if op == "add":
        result = gf2m_add(a, b)
    elif op == "mul":
        result = gf2m_mul(a, b, mod_poly)
    elif op == "inv":
        result = gf2m_inv(a, mod_poly)
    elif op == "pow":
        result = gf2m_pow(a, b, mod_poly)
    else:
        raise ValueError(f"unknown op: {op!r}, expected add/mul/inv/pow")

    return DescriptiveResult(
        name="gf2m_arithmetic",
        value=float(result),
        extra={
            "result": result,
            "a": a,
            "b": b,
            "m": m,
            "op": op,
            "mod_poly": mod_poly,
        },
    )


gf2m = gf2m_arithmetic


def cheatsheet() -> str:
    return "gf2m_arithmetic({}) -> GF(2^m) finite field arithmetic."
