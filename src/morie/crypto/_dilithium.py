"""ML-DSA / Dilithium-lite -- simplified lattice-based signature.

Educational reference capturing the core Dilithium construction:
MLWE keygen, rejection sampling sign, HighBits/LowBits verify.
Uses a simplified parameter set, NOT full FIPS 204 compliance.

NOT constant-time. Educational/research only.
"""

from __future__ import annotations

import hashlib
import os

from morie.crypto._poly_ring import poly_add, poly_ring_mul, poly_sub

Q_DIL = 8380417
N_DIL = 256
K_DIL = 4
L_DIL = 4
GAMMA1 = 1 << 17
GAMMA2 = (Q_DIL - 1) // 88
ETA_DIL = 2
BETA_DIL = 78
OMEGA_DIL = 80


def _mod_q(x: int) -> int:
    return x % Q_DIL


def _mod_pm(x: int, q: int) -> int:
    r = x % q
    if r > q // 2:
        r -= q
    return r


def _high_bits(r: int) -> int:
    r_pos = r % Q_DIL
    r0 = _mod_pm(r_pos, 2 * GAMMA2)
    return (r_pos - r0) // (2 * GAMMA2)


def _low_bits(r: int) -> int:
    r_pos = r % Q_DIL
    return _mod_pm(r_pos, 2 * GAMMA2)


def _sample_uniform(seed: bytes, n: int = N_DIL) -> list[int]:
    stream = hashlib.shake_128(seed).digest(n * 4)
    poly = []
    for i in range(n):
        val = int.from_bytes(stream[i * 4 : i * 4 + 4], "little") % Q_DIL
        poly.append(val)
    return poly


def _sample_short(seed: bytes, eta: int = ETA_DIL, n: int = N_DIL) -> list[int]:
    stream = hashlib.shake_256(seed).digest(n * 2)
    poly = []
    for i in range(n):
        val = stream[i * 2] % (2 * eta + 1)
        poly.append(_mod_q(val - eta))
    return poly


def _sample_gamma1(seed: bytes, n: int = N_DIL) -> list[int]:
    stream = hashlib.shake_256(seed).digest(n * 4)
    poly = []
    for i in range(n):
        val = int.from_bytes(stream[i * 4 : i * 4 + 4], "little") % (2 * GAMMA1)
        poly.append(val - GAMMA1)
    return poly


def mldsa_keygen() -> tuple[bytes, bytes]:
    """Generate ML-DSA key pair (simplified Dilithium).

    :return: (pk_bytes, sk_bytes).
    """
    seed = os.urandom(32)
    rho = hashlib.sha3_256(seed).digest()
    sigma = hashlib.sha3_256(seed + b"\x01").digest()

    A = []
    for i in range(K_DIL):
        row = []
        for j in range(L_DIL):
            poly = _sample_uniform(rho + bytes([i, j]))
            row.append(poly)
        A.append(row)

    s1 = [_sample_short(sigma + bytes([i])) for i in range(L_DIL)]
    s2 = [_sample_short(sigma + bytes([L_DIL + i])) for i in range(K_DIL)]

    t = []
    for i in range(K_DIL):
        ti = [0] * N_DIL
        for j in range(L_DIL):
            prod = poly_ring_mul(A[i][j], s1[j], Q_DIL, N_DIL)
            ti = poly_add(ti, prod, Q_DIL)
        ti = poly_add(ti, s2[i], Q_DIL)
        t.append(ti)

    import json

    pk_bytes = json.dumps({"rho": rho.hex(), "t": t}).encode()
    sk_bytes = json.dumps(
        {
            "rho": rho.hex(),
            "s1": s1,
            "s2": s2,
            "t": t,
            "sigma": sigma.hex(),
        }
    ).encode()

    return pk_bytes, sk_bytes


def mldsa_sign(message: bytes, sk_bytes: bytes) -> bytes:
    """Sign a message with ML-DSA (simplified Dilithium).

    :param message: Message to sign.
    :param sk_bytes: Secret key from mldsa_keygen().
    :return: Signature bytes.
    """
    import json

    sk = json.loads(sk_bytes)
    rho = bytes.fromhex(sk["rho"])
    s1 = sk["s1"]

    A = []
    for i in range(K_DIL):
        row = []
        for j in range(L_DIL):
            poly = _sample_uniform(rho + bytes([i, j]))
            row.append(poly)
        A.append(row)

    mu = hashlib.sha3_256(message).digest()
    nonce = 0

    for _ in range(1000):
        y = [_sample_gamma1(mu + nonce.to_bytes(4, "little") + bytes([i])) for i in range(L_DIL)]
        nonce += 1

        w = []
        for i in range(K_DIL):
            wi = [0] * N_DIL
            for j in range(L_DIL):
                prod = poly_ring_mul(A[i][j], y[j], Q_DIL, N_DIL)
                wi = poly_add(wi, prod, Q_DIL)
            w.append(wi)

        w1 = [[_high_bits(c) for c in wi] for wi in w]

        w1_bytes = str(w1).encode()
        c_hash = hashlib.sha3_256(mu + w1_bytes).digest()
        c_poly = [0] * N_DIL
        for i in range(min(60, N_DIL)):
            c_poly[i] = 1 if (c_hash[i % 32] >> (i % 8)) & 1 else 0

        z = []
        for j in range(L_DIL):
            cs1 = poly_ring_mul(c_poly, s1[j], Q_DIL, N_DIL)
            zj = poly_add(y[j], cs1, Q_DIL)
            z.append(zj)

        max_z = max(abs(_mod_pm(c, Q_DIL)) for zj in z for c in zj)
        if max_z >= GAMMA1 - BETA_DIL:
            continue

        sig_data = {"z": z, "c_hash": c_hash.hex()}
        return json.dumps(sig_data).encode()

    raise RuntimeError("signing failed after 1000 attempts")


def mldsa_verify(message: bytes, signature: bytes, pk_bytes: bytes) -> bool:
    """Verify an ML-DSA signature (simplified Dilithium).

    :param message: Original message.
    :param signature: Signature from mldsa_sign().
    :param pk_bytes: Public key from mldsa_keygen().
    :return: True if valid.
    """
    import json

    pk = json.loads(pk_bytes)
    sig = json.loads(signature)

    rho = bytes.fromhex(pk["rho"])
    t = pk["t"]
    z = sig["z"]
    c_hash = bytes.fromhex(sig["c_hash"])

    A = []
    for i in range(K_DIL):
        row = []
        for j in range(L_DIL):
            poly = _sample_uniform(rho + bytes([i, j]))
            row.append(poly)
        A.append(row)

    c_poly = [0] * N_DIL
    for i in range(min(60, N_DIL)):
        c_poly[i] = 1 if (c_hash[i % 32] >> (i % 8)) & 1 else 0

    w_prime = []
    for i in range(K_DIL):
        wi = [0] * N_DIL
        for j in range(L_DIL):
            prod = poly_ring_mul(A[i][j], z[j], Q_DIL, N_DIL)
            wi = poly_add(wi, prod, Q_DIL)
        ct = poly_ring_mul(c_poly, t[i], Q_DIL, N_DIL)
        wi = poly_sub(wi, ct, Q_DIL)
        w_prime.append(wi)

    w1_prime = [[_high_bits(c) for c in wi] for wi in w_prime]

    mu = hashlib.sha3_256(message).digest()
    w1_bytes = str(w1_prime).encode()
    c_check = hashlib.sha3_256(mu + w1_bytes).digest()

    max_z = max(abs(_mod_pm(c, Q_DIL)) for zj in z for c in zj)
    if max_z >= GAMMA1 - BETA_DIL:
        return False

    return c_check == c_hash
