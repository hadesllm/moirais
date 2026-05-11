# morie.fn — function file (hadesllm/morie)
"""LDPC parity-check matrix generation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def ldpc_generate(n: int = 20, rate: float = 0.5, col_weight: int = 3) -> DescriptiveResult:
    """Generate a random regular LDPC parity-check matrix.

    :param n: Codeword length.
    :param rate: Code rate (k/n).
    :param col_weight: Column weight.
    :return: DescriptiveResult with H matrix.
    """
    from morie.crypto._ecc import ldpc_generate as _gen

    result = _gen(n=n, rate=rate, col_weight=col_weight)
    return DescriptiveResult(
        name="ldpc_generate",
        value=float(result["n"]),
        extra=result,
    )


ldpcg = ldpc_generate


def cheatsheet() -> str:
    return "ldpc_generate({}) -> LDPC parity-check matrix generation."
