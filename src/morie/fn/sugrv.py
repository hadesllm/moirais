"""Supergravity action parameters."""

from __future__ import annotations

from ._containers import DescriptiveResult


def supergravity_action(
    d: int = 11,
    N: int = 1,
) -> DescriptiveResult:
    """Compute properties of d-dimensional N-extended supergravity.

    Supergravity is the low-energy limit of string/M-theory.
    Maximum dimension is 11 (N=1). Maximum supercharges = 32.

    Key properties:
    - Supercharges: :math:`Q = N \\times 2^{\\lfloor d/2 \\rfloor}`
    - Gravitino count: N

    :param d: Spacetime dimensions. Must be 2-11.
    :param N: Number of supersymmetries. Must be >= 1.
    :return: DescriptiveResult with SUGRA properties.
    """
    if d < 2 or d > 11:
        raise ValueError(f"Dimension must be 2-11, got {d}.")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}.")
    spinor_dim = 2 ** (d // 2)
    supercharges = N * spinor_dim
    if supercharges > 32:
        raise ValueError(f"Max 32 supercharges; N={N}, d={d} gives {supercharges}.")
    gravitini = N
    form_fields = []
    if d == 11 and N == 1:
        form_fields = ["C_3 (3-form)"]
    elif d == 10 and N == 2:
        form_fields = ["B_2", "C_1", "C_3"]
    return DescriptiveResult(
        name="supergravity_action",
        value=float(supercharges),
        extra={
            "d": d,
            "N": N,
            "supercharges": supercharges,
            "gravitini": gravitini,
            "spinor_dim": spinor_dim,
            "form_fields": form_fields,
        },
    )


def cheatsheet() -> str:
    return "supergravity_action(d, N) -> d-dimensional N-extended SUGRA"


sugrv = supergravity_action
