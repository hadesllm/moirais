# morie.fn -- function file (hadesllm/morie)
"""Polynomial ring operations mod (x^n+1) over Z_q."""

from __future__ import annotations

from ._containers import DescriptiveResult


def poly_ring_op(
    a: list[int],
    b: list[int],
    q: int = 3329,
    op: str = "mul",
) -> DescriptiveResult:
    """Polynomial arithmetic in Z_q[x]/(x^n+1).

    :param a: First polynomial (coefficient list, constant term first).
    :param b: Second polynomial.
    :param q: Modulus.
    :param op: Operation -- "add", "sub", "mul".
    :return: DescriptiveResult with result polynomial.
    """
    from morie.crypto._poly_ring import poly_add, poly_ring_mul, poly_sub

    n = max(len(a), len(b))
    a_pad = a + [0] * (n - len(a))
    b_pad = b + [0] * (n - len(b))

    if op == "add":
        result = poly_add(a_pad, b_pad, q)
    elif op == "sub":
        result = poly_sub(a_pad, b_pad, q)
    elif op == "mul":
        result = poly_ring_mul(a_pad, b_pad, q, n)
    else:
        raise ValueError(f"unknown op: {op!r}, expected add/sub/mul")

    return DescriptiveResult(
        name="poly_ring_op",
        value=float(len(result)),
        extra={"result": result, "a": a, "b": b, "q": q, "op": op, "degree": n},
    )


plyrn = poly_ring_op


def cheatsheet() -> str:
    return "poly_ring_op({}) -> Polynomial ring operations mod (x^n+1) over Z_q."
