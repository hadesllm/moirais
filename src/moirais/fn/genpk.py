# moirais.fn — function file (hadesllm/moirais)
"""Generator and parity-check matrix construction."""

from __future__ import annotations

from ._containers import DescriptiveResult


def gen_parity_check(n: int = 7, k: int = 4, code_type: str = "hamming") -> DescriptiveResult:
    """Generate G and H matrices for a linear code.

    :param n: Codeword length.
    :param k: Message length.
    :param code_type: "hamming" or "random".
    :return: DescriptiveResult with G and H matrices.
    """
    from moirais.crypto._ecc import gen_parity_check as _gen

    result = _gen(n, k, code_type=code_type)
    return DescriptiveResult(
        name="gen_parity_check",
        value=float(result["n"]),
        extra=result,
    )


genpk = gen_parity_check


def cheatsheet() -> str:
    return "gen_parity_check({}) -> Generator and parity-check matrix construction."
