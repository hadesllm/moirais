"""ML-KEM-768 (FIPS 203, formerly Kyber768) — pure-Python reference.

NOT constant-time. NOT for production secrets. Educational/research only.
Uses SHAKE-128/256 from hashlib (stdlib). No external dependencies.
"""

from __future__ import annotations

import hashlib
import os

Q = 3329
N = 256
K = 3
ETA1 = 2
ETA2 = 2
DU = 10
DV = 4

ZETAS = [1]
_g = 17
for _ in range(N - 1):
    ZETAS.append((ZETAS[-1] * _g) % Q)


def _mod_q(x: int) -> int:
    return x % Q


def _ntt(f: list[int]) -> list[int]:
    a = f[:]
    k = 1
    length = 128
    while length >= 2:
        start = 0
        while start < N:
            zeta = ZETAS[k]
            k += 1
            for j in range(start, start + length):
                t = _mod_q(zeta * a[j + length])
                a[j + length] = _mod_q(a[j] - t)
                a[j] = _mod_q(a[j] + t)
            start += 2 * length
        length //= 2
    return a


def _inv_ntt(f: list[int]) -> list[int]:
    a = f[:]
    k = 127
    length = 2
    while length <= 128:
        start = 0
        while start < N:
            zeta = ZETAS[k]
            k -= 1
            for j in range(start, start + length):
                t = a[j]
                a[j] = _mod_q(t + a[j + length])
                a[j + length] = _mod_q(zeta * _mod_q(a[j + length] - t))
            start += 2 * length
        length *= 2
    inv_n = pow(N, Q - 2, Q)
    return [_mod_q(x * inv_n) for x in a]


def _poly_mul_ntt(a: list[int], b: list[int]) -> list[int]:
    return [_mod_q(a[i] * b[i]) for i in range(N)]


def _poly_add(a: list[int], b: list[int]) -> list[int]:
    return [_mod_q(a[i] + b[i]) for i in range(N)]


def _poly_sub(a: list[int], b: list[int]) -> list[int]:
    return [_mod_q(a[i] - b[i]) for i in range(N)]


def _compress(x: int, d: int) -> int:
    return (((x << d) + Q // 2) // Q) % (1 << d)


def _decompress(x: int, d: int) -> int:
    return ((x * Q) + (1 << (d - 1))) >> d


def _cbd(eta: int, data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    coeffs = []
    for i in range(N):
        a = sum(bits[2 * i * eta + j] for j in range(eta))
        b = sum(bits[2 * i * eta + eta + j] for j in range(eta))
        coeffs.append(_mod_q(a - b))
    return coeffs


def _shake128(data: bytes, length: int) -> bytes:
    return hashlib.shake_128(data).digest(length)


def _shake256(data: bytes, length: int) -> bytes:
    return hashlib.shake_256(data).digest(length)


def _sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def _sha3_512(data: bytes) -> bytes:
    return hashlib.sha3_512(data).digest()


def _encode(poly: list[int], d: int) -> bytes:
    bits = []
    for c in poly:
        v = c % (1 << d)
        for j in range(d):
            bits.append((v >> j) & 1)
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        out[i // 8] |= b << (i % 8)
    return bytes(out)


def _decode(data: bytes, d: int) -> list[int]:
    bits = []
    for byte in data:
        for j in range(8):
            bits.append((byte >> j) & 1)
    poly = []
    for i in range(N):
        v = 0
        for j in range(d):
            idx = i * d + j
            if idx < len(bits):
                v |= bits[idx] << j
        poly.append(v % Q)
    return poly


def mlkem768_keygen() -> tuple[bytes, bytes]:
    """Generate ML-KEM-768 key pair.

    :return: (public_key, secret_key) as bytes.
    """
    d = os.urandom(32)
    rho, sigma = _sha3_512(d)[:32], _sha3_512(d)[32:]

    a_hat = [[None] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            seed = rho + bytes([j, i])
            stream = _shake128(seed, 3 * N)
            poly = []
            idx = 0
            while len(poly) < N and idx + 2 < len(stream):
                d1 = stream[idx] | ((stream[idx + 1] & 0x0F) << 8)
                d2 = (stream[idx + 1] >> 4) | (stream[idx + 2] << 4)
                idx += 3
                if d1 < Q:
                    poly.append(d1)
                if d2 < Q and len(poly) < N:
                    poly.append(d2)
            while len(poly) < N:
                poly.append(0)
            a_hat[i][j] = poly

    s = []
    for i in range(K):
        noise_bytes = _shake256(sigma + bytes([i]), 64 * ETA1)
        s.append(_ntt(_cbd(ETA1, noise_bytes)))

    e = []
    for i in range(K):
        noise_bytes = _shake256(sigma + bytes([K + i]), 64 * ETA1)
        e.append(_ntt(_cbd(ETA1, noise_bytes)))

    t_hat = []
    for i in range(K):
        acc = [0] * N
        for j in range(K):
            prod = _poly_mul_ntt(a_hat[i][j], s[j])
            acc = _poly_add(acc, prod)
        t_hat.append(_poly_add(acc, e[i]))

    pk_bytes = b""
    for i in range(K):
        pk_bytes += _encode(t_hat[i], 12)
    pk_bytes += rho

    sk_bytes = b""
    for i in range(K):
        sk_bytes += _encode(s[i], 12)

    pk_hash = _sha3_256(pk_bytes)
    sk_full = sk_bytes + pk_bytes + pk_hash + d

    return pk_bytes, sk_full


def mlkem768_encaps(pk: bytes) -> tuple[bytes, bytes]:
    """Encapsulate a shared secret using the public key.

    :param pk: Public key from mlkem768_keygen().
    :return: (ciphertext, shared_secret_32_bytes).
    """
    t_hat = []
    offset = 0
    for _i in range(K):
        chunk = pk[offset : offset + 384]
        t_hat.append(_decode(chunk, 12))
        offset += 384
    rho = pk[offset : offset + 32]

    m = os.urandom(32)
    m_hash = _sha3_256(m)
    pk_hash = _sha3_256(pk)
    kr = _sha3_512(m_hash + pk_hash)
    shared_secret = kr[:32]
    coins = kr[32:]

    a_hat = [[None] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            seed = rho + bytes([j, i])
            stream = _shake128(seed, 3 * N)
            poly = []
            idx = 0
            while len(poly) < N and idx + 2 < len(stream):
                d1 = stream[idx] | ((stream[idx + 1] & 0x0F) << 8)
                d2 = (stream[idx + 1] >> 4) | (stream[idx + 2] << 4)
                idx += 3
                if d1 < Q:
                    poly.append(d1)
                if d2 < Q and len(poly) < N:
                    poly.append(d2)
            while len(poly) < N:
                poly.append(0)
            a_hat[i][j] = poly

    r = []
    for i in range(K):
        noise_bytes = _shake256(coins + bytes([i]), 64 * ETA1)
        r.append(_ntt(_cbd(ETA1, noise_bytes)))

    e1 = []
    for i in range(K):
        noise_bytes = _shake256(coins + bytes([K + i]), 64 * ETA2)
        e1.append(_cbd(ETA2, noise_bytes))

    e2_bytes = _shake256(coins + bytes([2 * K]), 64 * ETA2)
    e2 = _cbd(ETA2, e2_bytes)

    u = []
    for i in range(K):
        acc = [0] * N
        for j in range(K):
            prod = _poly_mul_ntt(a_hat[j][i], r[j])
            acc = _poly_add(acc, prod)
        acc_time = _inv_ntt(acc)
        u.append(_poly_add(acc_time, e1[i]))

    v_acc = [0] * N
    for j in range(K):
        prod = _poly_mul_ntt(t_hat[j], r[j])
        v_acc = _poly_add(v_acc, prod)
    v_time = _inv_ntt(v_acc)
    v = _poly_add(v_time, e2)

    m_poly = _decode(m_hash, 1)
    m_decomp = [_decompress(c, 1) for c in m_poly]
    v = _poly_add(v, m_decomp)

    ct = b""
    for i in range(K):
        compressed = [_compress(c, DU) for c in u[i]]
        ct += _encode(compressed, DU)
    v_compressed = [_compress(c, DV) for c in v]
    ct += _encode(v_compressed, DV)

    return ct, shared_secret


def mlkem768_decaps(sk: bytes, ct: bytes) -> bytes:
    """Decapsulate the shared secret using the secret key.

    :param sk: Secret key from mlkem768_keygen().
    :param ct: Ciphertext from mlkem768_encaps().
    :return: 32-byte shared secret.
    """
    s_hat = []
    offset = 0
    for _i in range(K):
        chunk = sk[offset : offset + 384]
        s_hat.append(_decode(chunk, 12))
        offset += 384

    u = []
    ct_offset = 0
    for _i in range(K):
        chunk_len = N * DU // 8
        u_compressed = _decode(ct[ct_offset : ct_offset + chunk_len], DU)
        u.append([_decompress(c, DU) for c in u_compressed])
        ct_offset += chunk_len

    v_chunk_len = N * DV // 8
    v_compressed = _decode(ct[ct_offset : ct_offset + v_chunk_len], DV)
    v = [_decompress(c, DV) for c in v_compressed]

    u_ntt = [_ntt(ui) for ui in u]
    acc = [0] * N
    for j in range(K):
        prod = _poly_mul_ntt(s_hat[j], u_ntt[j])
        acc = _poly_add(acc, prod)
    su = _inv_ntt(acc)

    m_poly = _poly_sub(v, su)
    m_bits = [_compress(c, 1) for c in m_poly]
    m_recovered = _encode(m_bits, 1)

    pk_start = K * 384
    pk_end = pk_start + K * 384 + 32
    pk = sk[pk_start:pk_end]
    pk_hash = _sha3_256(pk)

    kr = _sha3_512(m_recovered[:32] + pk_hash)
    shared_secret = kr[:32]

    return shared_secret
