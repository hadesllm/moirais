# moirais.fn — function file (hadesllm/moirais)
"""Number Theoretic Transform (NTT) over Z_q."""

from __future__ import annotations

from ._containers import DescriptiveResult


def _find_nth_root(q: int, n: int) -> int | None:
    """Find a primitive n-th root of unity mod q (incomplete NTT)."""
    if q % n != 1:
        return None
    for g_candidate in range(2, min(q, 1000)):
        g = pow(g_candidate, (q - 1) // n, q)
        if g != 1 and pow(g, n, q) == 1:
            ok = True
            for d in range(1, n):
                if n % d == 0 and d < n and pow(g, d, q) == 1:
                    ok = False
                    break
            if ok:
                return g
    return None


def ntt_transform(poly: list[int], q: int = 3329, inverse: bool = False) -> DescriptiveResult:
    """Forward or inverse NTT of a polynomial in Z_q[x]/(x^n+1).

    :param poly: Coefficient list (length must be power of 2).
    :param q: Prime modulus (q ≡ 1 mod 2n for complete NTT, or q ≡ 1 mod n
        for incomplete/Kyber-style NTT).
    :param inverse: If True, compute inverse NTT.
    :return: DescriptiveResult with transformed polynomial.
    """
    from moirais.crypto._lattice_core import _find_ntt_root
    from moirais.crypto._poly_ring import build_zetas, inv_ntt, ntt

    n = len(poly)
    g = _find_ntt_root(q, n)
    if g is None:
        g = _find_nth_root(q, n)
    if g is None:
        raise ValueError(f"no NTT root found for q={q}, n={n}")

    zetas = build_zetas(q, n, g)
    result = inv_ntt(poly, q, n, zetas) if inverse else ntt(poly, q, n, zetas)

    return DescriptiveResult(
        name="ntt_transform",
        value=float(n),
        extra={"result": result, "q": q, "n": n, "inverse": inverse, "root": g},
    )


nttfn = ntt_transform


def cheatsheet() -> str:
    return "ntt_transform({}) -> Number Theoretic Transform (NTT) over Z_q."
