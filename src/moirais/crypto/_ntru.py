"""NTRU key exchange — pure-Python educational reference.

Implements NTRU encryption in Z_q[x]/(x^n - 1) with polynomial
inversion via Newton lifting.

NOT constant-time. Educational/research only.
"""

from __future__ import annotations

import os

from moirais.crypto._poly_ring import poly_inv_mod_prime, poly_ring_mul_xn_minus_1


def _sample_ternary(n: int, d: int) -> list[int]:
    """Sample a ternary polynomial with d coefficients = 1 and d = -1."""
    rng_bytes = os.urandom(n * 4)
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int.from_bytes(rng_bytes[i * 4 : i * 4 + 4], "little") % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]

    poly = [0] * n
    for i in range(d):
        poly[indices[i]] = 1
    for i in range(d, 2 * d):
        if i < n:
            poly[indices[i]] = -1
    return poly


def ntru_keygen(n: int = 167, q: int = 128, p: int = 3) -> dict:
    """Generate an NTRU key pair.

    :param n: Polynomial degree (should be prime).
    :param q: Large modulus (power of 2).
    :param p: Small modulus (typically 3).
    :return: dict with pk (h polynomial), sk (f, fp polynomials), params.
    """
    d = n // 3

    for _ in range(100):
        f = _sample_ternary(n, d)
        f[0] = (f[0] + 1) % q

        fp = poly_inv_mod_prime(f, p, n)
        if fp is None:
            continue

        fq = poly_inv_mod_prime(f, q, n)
        if fq is None:
            continue

        g = _sample_ternary(n, d)

        pfq = [(p * c) % q for c in fq]
        h = poly_ring_mul_xn_minus_1(pfq, g, q, n)

        return {
            "pk": h,
            "sk": {"f": f, "fp": fp},
            "params": {"n": n, "q": q, "p": p, "d": d},
        }

    raise RuntimeError("keygen failed — no invertible f found after 100 attempts")


def ntru_encrypt(message: list[int], pk: list[int], n: int = 167, q: int = 128) -> list[int]:
    """Encrypt a message polynomial with NTRU.

    :param message: Message polynomial (coefficients in {-1, 0, 1}).
    :param pk: Public key polynomial h.
    :param n: Polynomial degree.
    :param q: Modulus.
    :return: Ciphertext polynomial.
    """
    d = n // 3
    r = _sample_ternary(n, d)
    rh = poly_ring_mul_xn_minus_1(r, pk, q, n)
    c = [(rh[i] + message[i]) % q for i in range(n)]
    return c


def ntru_decrypt(ciphertext: list[int], sk: dict, n: int = 167, q: int = 128, p: int = 3) -> list[int]:
    """Decrypt an NTRU ciphertext.

    :param ciphertext: Ciphertext polynomial.
    :param sk: Secret key dict with f and fp.
    :param n: Polynomial degree.
    :param q: Large modulus.
    :param p: Small modulus.
    :return: Recovered message polynomial.
    """
    f = sk["f"]
    fp = sk["fp"]

    a = poly_ring_mul_xn_minus_1(f, ciphertext, q, n)
    a_centered = []
    for c in a:
        v = c % q
        if v > q // 2:
            v -= q
        a_centered.append(v % p)

    m = poly_ring_mul_xn_minus_1(fp, a_centered, p, n)
    m_centered = []
    for c in m:
        v = c % p
        if v > p // 2:
            v -= p
        m_centered.append(v)
    return m_centered
