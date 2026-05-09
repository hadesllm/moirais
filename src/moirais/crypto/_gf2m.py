"""GF(2^m) finite field and GF(2) matrix arithmetic for Goppa codes and McEliece."""

from __future__ import annotations


def _gf2_poly_degree(p: int) -> int:
    """Degree of a GF(2) polynomial represented as an integer."""
    if p == 0:
        return -1
    return p.bit_length() - 1


def _gf2_poly_mod(a: int, b: int) -> int:
    """Reduce polynomial a modulo b over GF(2)."""
    db = _gf2_poly_degree(b)
    if db < 0:
        raise ZeroDivisionError
    while True:
        da = _gf2_poly_degree(a)
        if da < db:
            return a
        a ^= b << (da - db)


def _gf2_poly_mul(a: int, b: int) -> int:
    """Carry-less (XOR) multiplication of two GF(2) polynomials."""
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result


def _gf2_poly_gcd(a: int, b: int) -> int:
    """GCD of two GF(2) polynomials via Euclidean algorithm."""
    while b:
        a, b = b, _gf2_poly_mod(a, b)
    return a


def _gf2_poly_powmod(base: int, exp: int, mod: int) -> int:
    """Square-and-multiply exponentiation in GF(2)[x] mod an irreducible."""
    result = 1
    base = _gf2_poly_mod(base, mod)
    while exp > 0:
        if exp & 1:
            result = _gf2_poly_mod(_gf2_poly_mul(result, base), mod)
        exp >>= 1
        base = _gf2_poly_mod(_gf2_poly_mul(base, base), mod)
    return result


def find_irreducible(m: int) -> int:
    """Find an irreducible polynomial of degree m over GF(2).

    Returns the polynomial as an integer where bit k represents the
    coefficient of x^k. Uses a lookup table for m <= 13 (Conway polynomials)
    and brute-force search with Rabin's irreducibility test for larger m.

    Reference: Galbraith, "Mathematics of Public Key Cryptography", Ch. 2.

    :param m: desired degree.
    :return: irreducible polynomial as integer.
    :raises ValueError: if no irreducible found (should not happen).
    """
    known = {
        2: 0b111,
        3: 0b1011,
        4: 0b10011,
        5: 0b100101,
        6: 0b1000011,
        7: 0b10000011,
        8: 0b100011101,
        9: 0b1000010001,
        10: 0b10000001001,
        11: 0b100000000101,
        12: 0b1000001010011,
        13: 0b10000000011011,
    }
    if m in known:
        return known[m]
    candidate = (1 << m) | 1
    while candidate < (1 << (m + 1)):
        if _is_irreducible(candidate, m):
            return candidate
        candidate += 2
    raise ValueError(f"no irreducible polynomial found for m={m}")


def _is_irreducible(poly: int, m: int) -> bool:
    """Rabin's irreducibility test for a degree-m polynomial over GF(2).

    Checks that gcd(x^{2^i} - x, poly) = 1 for i = 1..floor(m/2).
    """
    x = 0b10
    for _ in range(m // 2):
        x = _gf2_poly_powmod(x, 2, poly)
        g = _gf2_poly_gcd(x ^ 0b10, poly)
        if g != 1:
            return False
    return True


def gf2m_add(a: int, b: int) -> int:
    """Add two elements in GF(2^m).

    Addition in characteristic 2 is bitwise XOR.

    :param a: field element as integer.
    :param b: field element as integer.
    :return: a + b in GF(2^m).
    """
    return a ^ b


def gf2m_mul(a: int, b: int, mod_poly: int) -> int:
    """Multiply two elements in GF(2^m) modulo an irreducible polynomial.

    Carry-less multiplication followed by reduction mod the irreducible.

    Reference: Galbraith, "Mathematics of Public Key Cryptography", §2.9.

    :param a: field element as integer.
    :param b: field element as integer.
    :param mod_poly: irreducible polynomial (from find_irreducible).
    :return: a * b in GF(2^m).
    """
    return _gf2_poly_mod(_gf2_poly_mul(a, b), mod_poly)


def gf2m_inv(a: int, mod_poly: int) -> int:
    """Multiplicative inverse in GF(2^m) via Fermat's little theorem.

    Computes a^{2^m - 2} mod p(x), which equals a^{-1} since the
    multiplicative group has order 2^m - 1.

    :param a: nonzero field element.
    :param mod_poly: irreducible polynomial.
    :return: a^{-1} in GF(2^m).
    :raises ValueError: if a is zero.
    """
    if a == 0:
        raise ValueError("zero has no inverse")
    m = _gf2_poly_degree(mod_poly)
    return _gf2_poly_powmod(a, (1 << m) - 2, mod_poly)


def gf2m_pow(a: int, exp: int, mod_poly: int) -> int:
    """Exponentiation in GF(2^m) via square-and-multiply.

    :param a: base element.
    :param exp: non-negative integer exponent.
    :param mod_poly: irreducible polynomial.
    :return: a^exp in GF(2^m).
    """
    return _gf2_poly_powmod(a, exp, mod_poly)


def gf2_matrix_add(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Matrix addition over GF(2).

    Each element is 0 or 1. Addition is XOR.

    :param A: first matrix (list of rows).
    :param B: second matrix (list of rows).
    :return: A + B over GF(2).
    """
    rows = len(A)
    cols = len(A[0])
    return [[(A[i][j] ^ B[i][j]) for j in range(cols)] for i in range(rows)]


def gf2_matrix_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Matrix multiplication over GF(2).

    Uses XOR for addition and AND for multiplication, then reduces mod 2.

    :param A: m x n matrix (list of rows).
    :param B: n x p matrix (list of rows).
    :return: m x p product matrix over GF(2).
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0
            for k in range(n):
                s ^= A[i][k] & B[k][j]
            C[i][j] = s
    return C


def gf2_matrix_inv(A: list[list[int]]) -> list[list[int]] | None:
    """Matrix inverse over GF(2) via Gauss-Jordan elimination.

    Reference: Galbraith, "Mathematics of Public Key Cryptography", §2.3.

    :param A: n x n matrix (list of rows, entries 0 or 1).
    :return: inverse matrix over GF(2), or None if singular.
    :raises ValueError: if matrix is not square.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("matrix must be square")

    aug = [A[i][:] + [1 if j == i else 0 for j in range(n)] for i in range(n)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row][col]:
                pivot = row
                break
        if pivot is None:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(n):
            if row != col and aug[row][col]:
                aug[row] = [aug[row][j] ^ aug[col][j] for j in range(2 * n)]

    return [aug[i][n:] for i in range(n)]
