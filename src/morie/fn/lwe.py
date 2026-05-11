# morie.fn — function file (hadesllm/morie)
"""LWE (Learning With Errors) sample generation."""

from __future__ import annotations

from ._containers import CryptoResult


def lwe_sample(n: int = 64, m: int = 128, q: int = 3329, sigma: float = 3.2) -> CryptoResult:
    """Generate a Learning With Errors instance (A, b = As + e mod q).

    :param n: Secret vector dimension.
    :param m: Number of samples.
    :param q: Modulus.
    :param sigma: Gaussian noise standard deviation.
    :return: CryptoResult with A, b, s, e in ``extra``.
    """
    from morie.crypto._lattice_core import lwe_sample as _lwe

    result = _lwe(n=n, m=m, q=q, sigma=sigma)
    return CryptoResult(
        algorithm="LWE",
        operation="sample",
        success=True,
        extra={
            "A": result["A"],
            "b": result["b"],
            "s": result["s"],
            "e": result["e"],
            "n": n,
            "m": m,
            "q": q,
        },
    )


lwe = lwe_sample


def cheatsheet() -> str:
    return "lwe_sample({}) -> LWE (Learning With Errors) sample generation."
