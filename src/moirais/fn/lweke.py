# moirais.fn — function file (hadesllm/moirais)
"""LWE-based Diffie-Hellman key exchange."""

from __future__ import annotations

from ._containers import CryptoResult


def lwe_key_exchange(n: int = 64, q: int = 3329, sigma: float = 3.2) -> CryptoResult:
    """Simulate a Diffie-Hellman-style LWE key exchange.

    :param n: Matrix/vector dimension.
    :param q: Modulus.
    :param sigma: Gaussian noise standard deviation.
    :return: CryptoResult with alice_key, bob_key, match in ``extra``.
    """
    from moirais.crypto._lattice_core import lwe_key_exchange as _kex

    result = _kex(n=n, q=q, sigma=sigma)
    return CryptoResult(
        algorithm="LWE-KEX",
        operation="key_exchange",
        success=result["match"],
        extra=result,
    )


lweke = lwe_key_exchange


def cheatsheet() -> str:
    return "lwe_key_exchange({}) -> LWE-based Diffie-Hellman key exchange."
