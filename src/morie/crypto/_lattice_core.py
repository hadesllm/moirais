"""Core lattice algorithms: Gram-Schmidt, LLL, BKZ, Babai, LWE, RLWE, key exchange."""

from __future__ import annotations

import numpy as np

from morie.crypto._poly_ring import (
    build_zetas,
    inv_ntt,
    ntt,
    poly_add,
    poly_mul_ntt,
    poly_ring_mul,
)


def gram_schmidt(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt orthogonalization for a lattice basis.

    Computes the orthogonal projection vectors B* and the Gram-Schmidt
    coefficients mu such that B[i] = B*[i] + sum_{j<i} mu[i,j] * B*[j].

    Reference: Galbraith, "Mathematics of Public Key Cryptography", Algorithm 17.1.

    :param basis: n x m matrix (rows are basis vectors).
    :return: (orthogonal_basis, mu_coefficients) both as numpy arrays.
    """
    B = basis.astype(np.float64).copy()
    n = B.shape[0]
    B_star = np.zeros_like(B, dtype=np.float64)
    mu = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        B_star[i] = B[i].copy()
        for j in range(i):
            dot_bstar = float(np.dot(B_star[j], B_star[j]))
            if dot_bstar < 1e-15:
                mu[i, j] = 0.0
            else:
                mu[i, j] = float(np.dot(B[i], B_star[j])) / dot_bstar
            B_star[i] = B_star[i] - mu[i, j] * B_star[j]
        mu[i, i] = 1.0

    return B_star, mu


def lll_reduce(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL lattice basis reduction (Lenstra-Lenstra-Lovász, 1982).

    Produces a delta-LLL-reduced basis satisfying:
    1. Size-reduction: |mu[i,j]| <= 0.5 for all j < i.
    2. Lovász condition: ||B*[k]||^2 >= (delta - mu[k,k-1]^2) * ||B*[k-1]||^2.

    The first vector of the output has norm within 2^{(n-1)/2} of the
    shortest lattice vector.

    Reference: Galbraith, "Mathematics of Public Key Cryptography", Algorithm 17.2.

    :param basis: n x m matrix (rows are basis vectors).
    :param delta: Lovász parameter (0.25 < delta <= 1, default 0.75).
    :return: LLL-reduced basis as numpy array.
    """
    B = basis.astype(np.float64).copy()
    n = B.shape[0]

    B_star, mu = gram_schmidt(B)
    k = 1

    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                r = round(mu[k, j])
                B[k] -= r * B[j]
                B_star, mu = gram_schmidt(B)

        bstar_k_sq = float(np.dot(B_star[k], B_star[k]))
        bstar_km1_sq = float(np.dot(B_star[k - 1], B_star[k - 1]))

        if bstar_k_sq >= (delta - mu[k, k - 1] ** 2) * bstar_km1_sq:
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            B_star, mu = gram_schmidt(B)
            k = max(k - 1, 1)

    return B


def bkz_reduce(basis: np.ndarray, block_size: int = 20, max_tours: int = 10, delta: float = 0.99) -> np.ndarray:
    """Simplified BKZ (Block Korkin-Zolotarev) lattice basis reduction.

    Uses LLL as a subroutine applied to overlapping blocks. Real BKZ
    implementations use pruned enumeration for the SVP oracle; this
    educational version uses LLL on each block.

    Reference: Galbraith, "Mathematics of Public Key Cryptography", §17.2.

    :param basis: n x m matrix (rows are basis vectors).
    :param block_size: BKZ block size (larger = better reduction, slower).
    :param max_tours: maximum number of BKZ tours.
    :param delta: LLL parameter for sub-reductions.
    :return: BKZ-reduced basis.
    """
    B = lll_reduce(basis, delta)
    n = B.shape[0]
    beta = min(block_size, n)

    for _ in range(max_tours):
        changed = False
        for k in range(n - beta + 1):
            block = B[k : k + beta].copy()
            reduced_block = lll_reduce(block, delta)
            if not np.allclose(block, reduced_block, atol=1e-10):
                B[k : k + beta] = reduced_block
                changed = True
        B = lll_reduce(B, delta)
        if not changed:
            break

    return B


def babai_nearest_plane(basis: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Babai's nearest plane algorithm for the Closest Vector Problem.

    Produces an approximate solution to CVP by rounding Gram-Schmidt
    coefficients layer by layer from the top. Quality depends on the
    basis quality -- apply LLL first for better results.

    Reference: Galbraith, "Mathematics of Public Key Cryptography", Algorithm 18.1.

    :param basis: n x m matrix (rows are basis vectors).
    :param target: target vector (length m).
    :return: approximate closest lattice vector.
    """
    B = basis.astype(np.float64)
    n = B.shape[0]

    B_star, mu = gram_schmidt(B)

    b = target.astype(np.float64).copy()
    coeffs = np.zeros(n)
    for i in range(n - 1, -1, -1):
        dot_bstar = float(np.dot(B_star[i], B_star[i]))
        if dot_bstar < 1e-15:
            coeffs[i] = 0
        else:
            coeffs[i] = round(float(np.dot(b, B_star[i])) / dot_bstar)
        b = b - coeffs[i] * B[i]

    return (coeffs @ B).astype(target.dtype)


def svp_approx(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """Approximate Shortest Vector Problem via LLL reduction.

    Returns the shortest vector in the LLL-reduced basis, which is
    within a factor of 2^{(n-1)/2} of the true shortest vector.

    :param basis: n x m matrix (rows are basis vectors).
    :param delta: LLL parameter.
    :return: approximately shortest lattice vector.
    """
    reduced = lll_reduce(basis, delta)
    norms = np.linalg.norm(reduced, axis=1)
    return reduced[np.argmin(norms)]


def lwe_sample(n: int = 64, m: int = 128, q: int = 3329, sigma: float = 3.2) -> dict:
    """Generate a Learning With Errors (LWE) instance.

    Produces (A, b = A*s + e mod q) where s is the secret vector and
    e is a discrete Gaussian error vector. The LWE problem is to
    recover s given (A, b).

    Reference: Hoffstein, Pipher, Silverman, "An Introduction to
    Mathematical Cryptography", §7.8.

    :param n: secret dimension.
    :param m: number of samples.
    :param q: modulus.
    :param sigma: Gaussian noise standard deviation.
    :return: dict with A, b, s, e (s and e are secret witnesses).
    """
    rng = np.random.default_rng()
    A = rng.integers(0, q, size=(m, n))
    s = rng.integers(0, q, size=n)
    e = np.round(rng.normal(0, sigma, size=m)).astype(np.int64) % q
    b = (A @ s + e) % q
    return {"A": A, "b": b, "s": s, "e": e, "n": n, "m": m, "q": q}


def rlwe_sample(n: int = 256, q: int = 3329, sigma: float = 3.2) -> dict:
    """Generate a Ring-LWE instance in Z_q[x]/(x^n+1).

    The Ring-LWE problem is the ring analogue of LWE: given (a, b = a*s + e),
    recover the secret polynomial s. When q is NTT-friendly (q ≡ 1 mod 2n),
    multiplication uses NTT; otherwise falls back to schoolbook.

    Reference: Hoffstein, Pipher, Silverman, "An Introduction to
    Mathematical Cryptography", §7.10.

    :param n: polynomial degree (must be power of 2).
    :param q: prime modulus.
    :param sigma: Gaussian noise standard deviation.
    :return: dict with a, b, s, e polynomials and parameters.
    """
    rng = np.random.default_rng()
    a = [int(x) for x in rng.integers(0, q, size=n)]
    s = [int(x) % q for x in np.round(rng.normal(0, sigma, size=n)).astype(np.int64)]
    e = [int(x) % q for x in np.round(rng.normal(0, sigma, size=n)).astype(np.int64)]

    g = _find_ntt_root(q, n)
    if g is None:
        b = poly_ring_mul(a, s, q, n)
        b = poly_add(b, e, q)
        return {"a": a, "b": b, "s": s, "e": e, "n": n, "q": q, "ntt": False}

    zetas = build_zetas(q, n, g)
    a_ntt = ntt(a, q, n, zetas)
    s_ntt = ntt(s, q, n, zetas)
    as_ntt = poly_mul_ntt(a_ntt, s_ntt, q)
    as_poly = inv_ntt(as_ntt, q, n, zetas)
    b = poly_add(as_poly, e, q)
    return {"a": a, "b": b, "s": s, "e": e, "n": n, "q": q, "ntt": True}


def lwe_key_exchange(n: int = 64, q: int = 3329, sigma: float = 3.2) -> dict:
    """Simulate a Diffie-Hellman-style LWE key exchange.

    Alice and Bob each generate LWE instances with a shared matrix A.
    Alice sends b_A = s_A^T * A + e_A, Bob sends b_B = A * s_B + e_B.
    Both derive approximately equal shared keys via rounding.

    Reference: Hoffstein, Pipher, Silverman, "An Introduction to
    Mathematical Cryptography", §7.11.

    :param n: dimension of the shared matrix A.
    :param q: modulus.
    :param sigma: Gaussian noise standard deviation.
    :return: dict with alice_key, bob_key, match (bool), and intermediates.
    """
    rng = np.random.default_rng()
    A = rng.integers(0, q, size=(n, n))

    sa = rng.integers(0, q, size=n)
    ea = np.round(rng.normal(0, sigma, size=n)).astype(np.int64) % q
    ba = (sa @ A + ea) % q

    sb = rng.integers(0, q, size=n)
    eb = np.round(rng.normal(0, sigma, size=n)).astype(np.int64) % q
    bb = (A @ sb + eb) % q

    ka_raw = int((ba @ sb) % q)
    kb_raw = int((sa @ bb) % q)

    ka = ka_raw * 2 // q
    kb = kb_raw * 2 // q
    match = ka == kb

    return {
        "alice_key": ka,
        "bob_key": kb,
        "match": match,
        "ka_raw": ka_raw,
        "kb_raw": kb_raw,
        "n": n,
        "q": q,
    }


def _find_ntt_root(q: int, n: int) -> int | None:
    """Find a primitive 2n-th root of unity mod q for NTT.

    Requires q ≡ 1 mod 2n. The root g satisfies g^n ≡ -1 (mod q),
    which ensures the NTT butterfly structure is valid for the
    ring Z_q[x]/(x^n + 1).
    """
    if q % (2 * n) != 1:
        return None
    for g_candidate in range(2, min(q, 1000)):
        g = pow(g_candidate, (q - 1) // (2 * n), q)
        if g != 1 and pow(g, n, q) == q - 1:
            return g
    return None
