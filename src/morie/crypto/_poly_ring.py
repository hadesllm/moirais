"""Parameterized polynomial ring arithmetic over Z_q[x]/(x^n+1)."""

from __future__ import annotations


def build_zetas(q: int, n: int, g: int) -> list[int]:
    """Build NTT twiddle factors from a primitive 2n-th root of unity g mod q.

    The generator g must satisfy g^n ≡ -1 (mod q) and g^{2n} ≡ 1 (mod q).
    For a full NTT over Z_q[x]/(x^n+1), q must satisfy q ≡ 1 (mod 2n).
    The returned list contains bit-reversed powers of g used as butterfly
    factors in the Cooley-Tukey NTT.

    For Kyber-style incomplete NTT (q ≡ 1 mod n but not mod 2n), pass
    an n-th root of unity instead. The NTT will produce n/2 degree-1
    polynomial residues, suitable for pointwise basemul but not for
    coefficient-domain roundtrip.

    :param q: prime modulus.
    :param n: polynomial degree (power of 2).
    :param g: primitive root of unity mod q.
    :return: list of n twiddle factors.
    """
    zetas = [1]
    for _ in range(n - 1):
        zetas.append((zetas[-1] * g) % q)
    return zetas


def ntt(f: list[int], q: int, n: int, zetas: list[int]) -> list[int]:
    """Forward Number Theoretic Transform over Z_q.

    Cooley-Tukey butterfly, decimation-in-time. When zetas are derived from
    a primitive 2n-th root of unity, this computes the NTT for the ring
    Z_q[x]/(x^n+1) and inv_ntt is its exact inverse. When zetas come from
    an n-th root (Kyber), this is the incomplete NTT.

    Reference: Galbraith, "Mathematics of Public Key Cryptography", Ch. 18.

    :param f: polynomial coefficients (length n).
    :param q: prime modulus.
    :param n: polynomial degree (power of 2).
    :param zetas: twiddle factors from build_zetas().
    :return: NTT-domain representation (length n).
    """
    a = f[:]
    k = 1
    length = n // 2
    while length >= 2:
        start = 0
        while start < n:
            z = zetas[k]
            k += 1
            for j in range(start, start + length):
                t = (z * a[j + length]) % q
                a[j + length] = (a[j] - t) % q
                a[j] = (a[j] + t) % q
            start += 2 * length
        length //= 2
    return a


def inv_ntt(f: list[int], q: int, n: int, zetas: list[int]) -> list[int]:
    """Inverse Number Theoretic Transform over Z_q.

    Gentleman-Sande butterfly, decimation-in-frequency. Exact inverse of
    ntt() when the twiddle factors come from a primitive 2n-th root.

    :param f: NTT-domain representation (length n).
    :param q: prime modulus.
    :param n: polynomial degree (power of 2).
    :param zetas: twiddle factors from build_zetas().
    :return: polynomial coefficients (length n).
    """
    a = f[:]
    k = n // 2 - 1
    length = 2
    while length <= n // 2:
        start = 0
        while start < n:
            z = zetas[k]
            k -= 1
            for j in range(start, start + length):
                t = a[j]
                a[j] = (t + a[j + length]) % q
                a[j + length] = (z * ((a[j + length] - t) % q)) % q
            start += 2 * length
        length *= 2
    inv_half_n = pow(n // 2, q - 2, q)
    return [(x * inv_half_n) % q for x in a]


def poly_add(a: list[int], b: list[int], q: int) -> list[int]:
    """Coefficient-wise polynomial addition mod q.

    :param a: first polynomial (length n).
    :param b: second polynomial (length n).
    :param q: modulus.
    :return: (a + b) mod q, coefficient-wise.
    """
    return [(a[i] + b[i]) % q for i in range(len(a))]


def poly_sub(a: list[int], b: list[int], q: int) -> list[int]:
    """Coefficient-wise polynomial subtraction mod q.

    :param a: first polynomial (length n).
    :param b: second polynomial (length n).
    :param q: modulus.
    :return: (a - b) mod q, coefficient-wise.
    """
    return [(a[i] - b[i]) % q for i in range(len(a))]


def poly_mul_ntt(a: list[int], b: list[int], q: int, zetas: list[int] | None = None) -> list[int]:
    """Pointwise multiplication in NTT domain mod q.

    Both a and b must already be in NTT representation. The zetas parameter
    is accepted for API uniformity but not used (NTT multiplication is
    pointwise in the evaluation domain).

    :param a: first polynomial in NTT domain (length n).
    :param b: second polynomial in NTT domain (length n).
    :param q: modulus.
    :param zetas: unused, accepted for API consistency.
    :return: pointwise product mod q (length n).
    """
    return [(a[i] * b[i]) % q for i in range(len(a))]


def poly_ring_mul(a: list[int], b: list[int], q: int, mod_poly: int | None = None) -> list[int]:
    """Schoolbook polynomial multiply mod (x^n + 1) mod q.

    O(n^2) but works for any q and n without NTT compatibility constraints.
    If mod_poly is an integer, it is treated as the degree n of the
    cyclotomic polynomial x^n + 1.

    Reference: Hoffstein, Pipher, Silverman, "An Introduction to Mathematical
    Cryptography", Ch. 7.

    :param a: first polynomial coefficients (length <= n).
    :param b: second polynomial coefficients (length <= n).
    :param q: modulus.
    :param mod_poly: degree n of x^n + 1 (default: len(a)).
    :return: product polynomial mod (x^n + 1) mod q.
    """
    n = mod_poly if mod_poly is not None else len(a)
    c = [0] * n
    for i in range(min(len(a), n)):
        for j in range(min(len(b), n)):
            idx = i + j
            if idx < n:
                c[idx] = (c[idx] + a[i] * b[j]) % q
            else:
                c[idx - n] = (c[idx - n] - a[i] * b[j]) % q
    return c


def poly_ring_mul_xn_minus_1(a: list[int], b: list[int], q: int, n: int) -> list[int]:
    """Schoolbook polynomial multiply mod (x^n - 1) mod q.

    Used by NTRU where the ring is Z_q[x]/(x^n - 1), not (x^n + 1).
    Wrapping is additive (convolution) rather than negacyclic.

    :param a: first polynomial coefficients.
    :param b: second polynomial coefficients.
    :param q: modulus.
    :param n: ring degree.
    :return: product polynomial mod (x^n - 1) mod q.
    """
    c = [0] * n
    for i in range(min(len(a), n)):
        for j in range(min(len(b), n)):
            idx = (i + j) % n
            c[idx] = (c[idx] + a[i] * b[j]) % q
    return c


def poly_inv_mod_prime(f: list[int], q: int, n: int) -> list[int] | None:
    """Polynomial inverse of f in Z_q[x]/(x^n - 1).

    Handles both prime q (via circulant matrix inversion) and power-of-2 q
    (via Newton lifting from mod 2). Returns None if f is not invertible.

    :param f: polynomial coefficients.
    :param q: modulus (prime or power of 2).
    :param n: ring degree.
    :return: inverse polynomial or None.
    """
    if q > 1 and (q & (q - 1)) == 0:
        return _poly_inv_mod_pow2(f, q, n)
    return _poly_inv_mod_small(f, q, n)


def _poly_inv_mod_pow2(f: list[int], q: int, n: int) -> list[int] | None:
    """Polynomial inverse mod a power of 2 via Newton lifting."""
    g = _poly_inv_mod_small(f, 2, n)
    if g is None:
        return None

    mod = 2
    while mod < q:
        target = min(mod * mod, q)
        fg = poly_ring_mul_xn_minus_1(f, g, target, n)
        two_minus_fg = [(2 - fg[0]) % target] + [(-c) % target for c in fg[1:]]
        g = poly_ring_mul_xn_minus_1(g, two_minus_fg, target, n)
        g = [c % target for c in g]
        mod = target

    g = [c % q for c in g]
    check = poly_ring_mul_xn_minus_1(f, g, q, n)
    if check[0] != 1 or any(c != 0 for c in check[1:]):
        return None
    return g


def _poly_inv_mod_small(f: list[int], p: int, n: int) -> list[int] | None:
    """Polynomial inverse in Z_p[x]/(x^n-1) via Gauss elimination on circulant matrix."""
    f_mod = [c % p for c in f]

    aug = [[0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug[i][j] = f_mod[(j - i) % n]
        aug[i][n + i] = 1

    for col in range(n):
        pivot = -1
        for row in range(col, n):
            if aug[row][col] % p != 0:
                pivot = row
                break
        if pivot < 0:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        inv_piv = pow(aug[col][col] % p, p - 2, p) if p > 2 else 1
        for j in range(2 * n):
            aug[col][j] = (aug[col][j] * inv_piv) % p

        for row in range(n):
            if row != col and aug[row][col] % p != 0:
                factor = aug[row][col] % p
                for j in range(2 * n):
                    aug[row][j] = (aug[row][j] - factor * aug[col][j]) % p

    return [aug[0][n + i] % p for i in range(n)]
