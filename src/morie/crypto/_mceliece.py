"""Classic McEliece — syndrome-based KEM (simplified).

Educational reference for the McEliece cryptosystem using
binary Goppa codes. Uses simplified parameters.

NOT constant-time. Educational/research only.
"""

from __future__ import annotations

import hashlib

import numpy as np

from morie.crypto._ecc import goppa_generate


def mceliece_keygen(m: int = 4, t: int = 2) -> dict:
    """Generate a simplified McEliece key pair.

    :param m: GF(2^m) extension degree.
    :param t: Error-correcting capability.
    :return: dict with pk (public matrix T), sk (decoder info), params.
    """
    code = goppa_generate(m=m, t=t)
    H = code["H"]
    n = code["n"]
    k = code["k"]

    rng = np.random.default_rng()
    perm = rng.permutation(n).astype(int)
    H_perm = H[:, perm]

    return {
        "pk": H_perm,
        "sk": {"H": H, "perm": perm, "inv_perm": np.argsort(perm), "m": m, "t": t},
        "params": {"n": n, "k": k, "m": m, "t": t},
    }


def mceliece_encaps(pk: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Encapsulate a key using McEliece.

    :param pk: Public parity-check matrix (permuted).
    :return: (shared_secret, ciphertext_syndrome).
    """
    r, n = pk.shape

    rng = np.random.default_rng()
    e = np.zeros(n, dtype=np.uint8)
    error_positions = rng.choice(n, size=min(2, n), replace=False)
    e[error_positions] = 1

    syndrome = (pk @ e) % 2

    e_bytes = bytes(e.tolist())
    shared_secret = hashlib.sha3_256(b"\x01" + e_bytes + bytes(syndrome.tolist())).digest()

    return shared_secret, syndrome


def mceliece_decaps(syndrome: np.ndarray, sk: dict) -> bytes:
    """Decapsulate a McEliece ciphertext.

    :param syndrome: Syndrome from mceliece_encaps().
    :param sk: Secret key from mceliece_keygen().
    :return: 32-byte shared secret.
    """
    H = sk["H"]
    inv_perm = sk["inv_perm"]
    n = H.shape[1]

    s_unperm = syndrome

    e_guess = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        col = H[:, i]
        if np.array_equal((col % 2), (s_unperm % 2)):
            e_guess[i] = 1
            break

    e_original = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        e_original[inv_perm[i]] = e_guess[i]

    e_bytes = bytes(e_original.tolist())
    shared_secret = hashlib.sha3_256(b"\x01" + e_bytes + bytes(syndrome.tolist())).digest()

    return shared_secret
