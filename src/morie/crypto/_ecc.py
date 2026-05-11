"""Error-correcting codes: Hamming, Goppa, LDPC.

Provides encode/decode primitives used by McEliece and as standalone
educational tools for understanding code-based cryptography.

NOT constant-time. Educational/research only.
"""

from __future__ import annotations

import numpy as np


def hamming_generator(r: int) -> tuple[np.ndarray, np.ndarray]:
    """Build systematic Hamming(2^r - 1, 2^r - 1 - r) generator and parity-check matrices.

    :param r: Parity bits (r >= 2).
    :return: (G, H) where G is k x n and H is r x n, all over GF(2).
    """
    n = (1 << r) - 1
    k = n - r

    H_cols = []
    for i in range(1, n + 1):
        col = [(i >> bit) & 1 for bit in range(r)]
        H_cols.append(col)

    H = np.array(H_cols, dtype=np.uint8).T

    parity_cols = []
    info_cols = []
    for j in range(n):
        col = H[:, j]
        weight = int(np.sum(col))
        if weight == 1 and int(col[int(np.argmax(col))]) == 1:
            parity_cols.append(j)
        else:
            info_cols.append(j)

    P_info = H[:, info_cols]
    H_sys = np.hstack([P_info, np.eye(r, dtype=np.uint8)])
    G_sys = np.hstack([np.eye(k, dtype=np.uint8), P_info.T])

    return G_sys.astype(np.uint8), H_sys.astype(np.uint8)


def hamming_encode(data: np.ndarray, r: int = 3) -> dict:
    """Encode a message using a Hamming code.

    :param data: Binary message vector (length k = 2^r - 1 - r).
    :param r: Parity bits.
    :return: dict with codeword, G, H, n, k.
    """
    G, H = hamming_generator(r)
    k = G.shape[0]
    msg = np.asarray(data, dtype=np.uint8).flatten()[:k]
    if len(msg) < k:
        msg = np.pad(msg, (0, k - len(msg)))
    codeword = (msg @ G) % 2
    return {"codeword": codeword, "G": G, "H": H, "n": G.shape[1], "k": k}


def hamming_decode(received: np.ndarray, r: int = 3) -> dict:
    """Decode and correct a Hamming codeword.

    :param received: Received binary vector (length n = 2^r - 1).
    :param r: Parity bits.
    :return: dict with corrected, message, syndrome, error_pos.
    """
    G, H = hamming_generator(r)
    n = H.shape[1]
    k = G.shape[0]
    w = np.asarray(received, dtype=np.uint8).flatten()[:n]

    syndrome = (H @ w) % 2
    error_pos = -1

    if np.any(syndrome):
        s_val = 0
        for i, bit in enumerate(syndrome):
            s_val |= int(bit) << i
        if 1 <= s_val <= n:
            error_pos = s_val - 1
            w = w.copy()
            w[error_pos] ^= 1

    message = w[:k]
    return {
        "corrected": w,
        "message": message,
        "syndrome": syndrome,
        "error_pos": error_pos,
        "success": True,
    }


def syndrome_compute(H: np.ndarray, received: np.ndarray) -> np.ndarray:
    """Compute the syndrome s = H @ received mod 2.

    :param H: Parity-check matrix (r x n) over GF(2).
    :param received: Received binary vector.
    :return: Syndrome vector.
    """
    w = np.asarray(received, dtype=np.uint8).flatten()
    return (np.asarray(H, dtype=np.uint8) @ w) % 2


def gen_parity_check(n: int, k: int, code_type: str = "hamming") -> dict:
    """Generate generator and parity-check matrices.

    :param n: Codeword length.
    :param k: Message length.
    :param code_type: "hamming" or "random".
    :return: dict with G, H matrices.
    """
    r = n - k
    if code_type == "hamming":
        import math

        r_ham = max(2, int(math.ceil(math.log2(n + 1))))
        G, H = hamming_generator(r_ham)
        return {"G": G, "H": H, "n": G.shape[1], "k": G.shape[0], "type": code_type}

    rng = np.random.default_rng()
    P = rng.integers(0, 2, size=(k, r), dtype=np.uint8)
    G = np.hstack([np.eye(k, dtype=np.uint8), P])
    H = np.hstack([P.T, np.eye(r, dtype=np.uint8)])
    return {"G": G, "H": H, "n": n, "k": k, "type": code_type}


def ldpc_generate(n: int = 20, rate: float = 0.5, col_weight: int = 3) -> dict:
    """Generate a random regular LDPC parity-check matrix.

    :param n: Codeword length.
    :param rate: Code rate (k/n).
    :param col_weight: Column weight (number of 1s per column).
    :return: dict with H, G (approximate), n, k.
    """
    k = int(n * rate)
    m = n - k

    H = np.zeros((m, n), dtype=np.uint8)
    rng = np.random.default_rng()
    for j in range(n):
        rows = rng.choice(m, size=min(col_weight, m), replace=False)
        H[rows, j] = 1

    return {"H": H, "n": n, "k": k, "m": m, "col_weight": col_weight}


def ldpc_encode(G: np.ndarray, message: np.ndarray) -> dict:
    """Encode using a generator matrix (general linear code).

    :param G: Generator matrix (k x n) over GF(2).
    :param message: Binary message vector (length k).
    :return: dict with codeword.
    """
    G = np.asarray(G, dtype=np.uint8)
    msg = np.asarray(message, dtype=np.uint8).flatten()
    codeword = (msg @ G) % 2
    return {"codeword": codeword, "n": G.shape[1], "k": G.shape[0]}


def ldpc_decode(H: np.ndarray, received: np.ndarray, max_iter: int = 50) -> dict:
    """LDPC bit-flipping decoder.

    :param H: Parity-check matrix (m x n) over GF(2).
    :param received: Received binary vector (possibly corrupted).
    :param max_iter: Maximum decoding iterations.
    :return: dict with decoded, syndrome, iterations, success.
    """
    H = np.asarray(H, dtype=np.uint8)
    x = np.asarray(received, dtype=np.uint8).flatten().copy()
    m, n = H.shape

    for iteration in range(1, max_iter + 1):
        s = (H @ x) % 2
        if not np.any(s):
            return {
                "decoded": x,
                "syndrome": s,
                "iterations": iteration,
                "success": True,
            }

        unsat = np.zeros(n, dtype=int)
        for i in range(m):
            if s[i]:
                for j in range(n):
                    if H[i, j]:
                        unsat[j] += 1

        max_unsat = np.max(unsat)
        if max_unsat == 0:
            break
        flip_candidates = np.where(unsat == max_unsat)[0]
        for j in flip_candidates:
            x[j] ^= 1

    s = (H @ x) % 2
    return {
        "decoded": x,
        "syndrome": s,
        "iterations": max_iter,
        "success": not np.any(s),
    }


def goppa_generate(m: int = 4, t: int = 2) -> dict:
    """Generate a binary Goppa code (simplified educational version).

    :param m: GF(2^m) extension degree (field size = 2^m).
    :param t: Error-correcting capability.
    :return: dict with H (parity-check), n, k, t.
    """
    from morie.crypto._gf2m import find_irreducible, gf2m_inv, gf2m_mul

    mod_poly = find_irreducible(m)
    field_size = 1 << m

    support = list(range(field_size))
    n = len(support)
    k = n - m * t

    rng = np.random.default_rng()
    g_roots = rng.choice(field_size, size=min(t, field_size), replace=False).tolist()

    H_rows = []
    for i in range(min(t, field_size)):
        row = []
        for alpha in support:
            diff = alpha ^ g_roots[i]
            if diff == 0:
                row.append(0)
            else:
                inv_val = gf2m_inv(diff, mod_poly)
                val = (
                    gf2m_mul(1, inv_val, mod_poly) if i == 0 else gf2m_mul(pow(alpha, i, field_size), inv_val, mod_poly)
                )
                row.append(val & 1)
        H_rows.append(row)

    H = np.array(H_rows, dtype=np.uint8) if H_rows else np.zeros((0, n), dtype=np.uint8)

    return {"H": H, "n": n, "k": max(k, 0), "t": t, "m": m, "field_size": field_size}
