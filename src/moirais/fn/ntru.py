# moirais.fn — function file (hadesllm/moirais)
"""NTRU post-quantum key exchange — keygen."""

from __future__ import annotations

from ._containers import CryptoResult


def ntru_keygen(n: int = 167, q: int = 128) -> CryptoResult:
    """Generate an NTRU key pair.

    :param n: Polynomial degree (prime recommended).
    :param q: Modulus (power of 2).
    :return: CryptoResult with pk and sk in ``extra``.
    """
    from moirais.crypto._ntru import ntru_keygen as _keygen

    result = _keygen(n=n, q=q)
    return CryptoResult(
        algorithm="NTRU",
        operation="keygen",
        success=True,
        extra=result,
    )


ntru = ntru_keygen


def cheatsheet() -> str:
    return "ntru_keygen({}) -> NTRU post-quantum key exchange — keygen."
