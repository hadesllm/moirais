# morie.fn -- function file (hadesllm/morie)
"""Binary Goppa code construction."""

from __future__ import annotations

from ._containers import CryptoResult


def goppa_code(m: int = 4, t: int = 2) -> CryptoResult:
    """Generate a binary Goppa code.

    :param m: GF(2^m) extension degree.
    :param t: Error-correcting capability.
    :return: CryptoResult with parity-check matrix H in ``extra``.
    """
    from morie.crypto._ecc import goppa_generate

    result = goppa_generate(m=m, t=t)
    return CryptoResult(
        algorithm="Goppa",
        operation="generate",
        success=True,
        extra=result,
    )


goppa = goppa_code


def cheatsheet() -> str:
    return "goppa_code({}) -> Binary Goppa code construction."
