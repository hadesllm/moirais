# morie.fn — function file (hadesllm/morie)
"""Ring-LWE key generation in Z_q[x]/(x^n+1)."""

from __future__ import annotations

from ._containers import CryptoResult


def rlwe_keygen(n: int = 256, q: int = 3329, sigma: float = 3.2) -> CryptoResult:
    """Generate a Ring-LWE instance (public polynomial pair).

    :param n: Polynomial degree (power of 2).
    :param q: Modulus.
    :param sigma: Gaussian noise standard deviation.
    :return: CryptoResult with a, b, s, e polynomials in ``extra``.
    """
    from morie.crypto._lattice_core import rlwe_sample as _rlwe

    result = _rlwe(n=n, q=q, sigma=sigma)
    return CryptoResult(
        algorithm="Ring-LWE",
        operation="keygen",
        success=True,
        extra=result,
    )


rlwe = rlwe_keygen


def cheatsheet() -> str:
    return "rlwe_keygen({}) -> Ring-LWE key generation in Z_q[x]/(x^n+1)."
