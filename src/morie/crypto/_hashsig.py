"""Hash-based signatures: Lamport OTS, Winternitz OTS, Merkle trees.

Uses stdlib hashlib (SHA-3/SHAKE). No external dependencies.
NOT constant-time. Educational/research only.
"""

from __future__ import annotations

import hashlib
import os


def _sha256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def _hash_n(data: bytes, n: int = 32) -> bytes:
    return hashlib.shake_256(data).digest(n)


def lamport_keygen(n: int = 256) -> tuple[list[list[bytes]], list[list[bytes]]]:
    """Generate a Lamport one-time signature key pair.

    :param n: Number of bits to sign (default: 256 for SHA-256 digest).
    :return: (sk, pk) where sk[i][b] and pk[i][b] are 32-byte values.
    """
    sk = [[os.urandom(32) for _ in range(2)] for _ in range(n)]
    pk = [[_sha256(sk[i][b]) for b in range(2)] for i in range(n)]
    return sk, pk


def lamport_sign(message: bytes, sk: list[list[bytes]]) -> list[bytes]:
    """Sign a message with a Lamport private key.

    :param message: Message bytes (will be hashed to n bits).
    :param sk: Private key from lamport_keygen().
    :return: Signature (list of n 32-byte values).
    """
    digest = _sha256(message)
    n = len(sk)
    bits = []
    for byte in digest:
        for i in range(8):
            bits.append((byte >> i) & 1)
    bits = bits[:n]
    return [sk[i][bits[i]] for i in range(n)]


def lamport_verify(message: bytes, signature: list[bytes], pk: list[list[bytes]]) -> bool:
    """Verify a Lamport signature.

    :param message: Original message bytes.
    :param signature: Signature from lamport_sign().
    :param pk: Public key from lamport_keygen().
    :return: True if valid.
    """
    digest = _sha256(message)
    n = len(pk)
    bits = []
    for byte in digest:
        for i in range(8):
            bits.append((byte >> i) & 1)
    bits = bits[:n]
    return all(_sha256(signature[i]) == pk[i][bits[i]] for i in range(n))


def wots_keygen(n: int = 32, w: int = 16) -> tuple[list[bytes], list[bytes]]:
    """Generate a WOTS+ key pair.

    :param n: Hash output length in bytes.
    :param w: Winternitz parameter (chain length = w).
    :return: (sk, pk) lists of n-byte values.
    """
    import math

    l1 = math.ceil(8 * n / math.log2(w))
    l2 = max(1, math.floor(math.log2(l1 * (w - 1)) / math.log2(w)) + 1)
    total_len = l1 + l2

    sk = [os.urandom(n) for _ in range(total_len)]
    pk = [_chain(sk[i], w - 1, n) for i in range(total_len)]
    return sk, pk


def wots_sign(message: bytes, sk: list[bytes], w: int = 16, n: int = 32) -> list[bytes]:
    """Sign with WOTS+.

    :param message: Message bytes (will be hashed).
    :param sk: Private key from wots_keygen().
    :param w: Winternitz parameter.
    :param n: Hash length.
    :return: Signature (list of n-byte values).
    """
    import math

    digest = _hash_n(message, n)
    l1 = math.ceil(8 * n / math.log2(w))

    msg_vals = _base_w(digest, w, l1)
    checksum = sum(w - 1 - v for v in msg_vals)

    l2 = max(1, math.floor(math.log2(l1 * (w - 1)) / math.log2(w)) + 1)
    cs_bytes = checksum.to_bytes(max(1, (checksum.bit_length() + 7) // 8), "big")
    cs_vals = _base_w(cs_bytes, w, l2)

    vals = msg_vals + cs_vals
    return [_chain(sk[i], vals[i], n) for i in range(len(vals))]


def wots_verify(message: bytes, signature: list[bytes], pk: list[bytes], w: int = 16, n: int = 32) -> bool:
    """Verify a WOTS+ signature.

    :param message: Original message.
    :param signature: Signature from wots_sign().
    :param pk: Public key from wots_keygen().
    :param w: Winternitz parameter.
    :param n: Hash length.
    :return: True if valid.
    """
    import math

    digest = _hash_n(message, n)
    l1 = math.ceil(8 * n / math.log2(w))

    msg_vals = _base_w(digest, w, l1)
    checksum = sum(w - 1 - v for v in msg_vals)

    l2 = max(1, math.floor(math.log2(l1 * (w - 1)) / math.log2(w)) + 1)
    cs_bytes = checksum.to_bytes(max(1, (checksum.bit_length() + 7) // 8), "big")
    cs_vals = _base_w(cs_bytes, w, l2)

    vals = msg_vals + cs_vals
    for i in range(len(vals)):
        remaining = w - 1 - vals[i]
        computed_pk = _chain(signature[i], remaining, n)
        if computed_pk != pk[i]:
            return False
    return True


def _chain(value: bytes, steps: int, n: int) -> bytes:
    for _ in range(steps):
        value = _hash_n(value, n)
    return value


def _base_w(data: bytes, w: int, out_len: int) -> list[int]:
    import math

    log_w = int(math.log2(w))
    vals = []
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    for i in range(out_len):
        v = 0
        for j in range(log_w):
            idx = i * log_w + j
            if idx < len(bits):
                v = (v << 1) | bits[idx]
            else:
                v <<= 1
        vals.append(v % w)
    return vals


def merkle_tree_build(leaves: list[bytes]) -> dict:
    """Build a Merkle tree from leaf values.

    :param leaves: List of leaf byte values (will be hashed).
    :return: dict with root, tree (list of levels), leaf_count.
    """
    hashed = [_sha256(leaf) for leaf in leaves]
    if len(hashed) == 0:
        return {"root": b"\x00" * 32, "tree": [], "leaf_count": 0}

    n = 1
    while n < len(hashed):
        n *= 2
    while len(hashed) < n:
        hashed.append(b"\x00" * 32)

    tree = [hashed]
    current = hashed
    while len(current) > 1:
        parent = []
        for i in range(0, len(current), 2):
            parent.append(_sha256(current[i] + current[i + 1]))
        tree.append(parent)
        current = parent

    return {"root": tree[-1][0], "tree": tree, "leaf_count": len(leaves)}


def merkle_auth_path(tree: list[list[bytes]], index: int) -> list[bytes]:
    """Generate an authentication path for a leaf in a Merkle tree.

    :param tree: Tree from merkle_tree_build().
    :param index: Leaf index.
    :return: List of sibling hashes along the path to root.
    """
    path = []
    idx = index
    for level in tree[:-1]:
        sibling = idx ^ 1
        if sibling < len(level):
            path.append(level[sibling])
        else:
            path.append(b"\x00" * 32)
        idx //= 2
    return path


def merkle_verify(leaf: bytes, index: int, auth_path: list[bytes], root: bytes) -> bool:
    """Verify a Merkle authentication path.

    :param leaf: Leaf value (will be hashed).
    :param index: Leaf index.
    :param auth_path: Authentication path from merkle_auth_path().
    :param root: Expected root hash.
    :return: True if valid.
    """
    current = _sha256(leaf)
    idx = index
    for sibling in auth_path:
        current = _sha256(current + sibling) if idx % 2 == 0 else _sha256(sibling + current)
        idx //= 2
    return current == root


def xmss_keygen(tree_height: int = 4, w: int = 16, n: int = 32) -> dict:
    """Generate an XMSS key pair (simplified).

    :param tree_height: Height of the Merkle tree (2^h one-time keys).
    :param w: WOTS Winternitz parameter.
    :param n: Hash length.
    :return: dict with pk (root), sk (seed + wots keys), tree.
    """
    num_keys = 1 << tree_height
    seed = os.urandom(32)

    wots_keys = []
    wots_pks = []
    for _i in range(num_keys):
        sk_i, pk_i = wots_keygen(n, w)
        wots_keys.append(sk_i)
        pk_bytes = b"".join(pk_i)
        wots_pks.append(pk_bytes)

    tree_data = merkle_tree_build(wots_pks)
    return {
        "pk": tree_data["root"],
        "sk": {"seed": seed, "wots_keys": wots_keys, "wots_pks": wots_pks, "index": 0},
        "tree": tree_data["tree"],
        "height": tree_height,
        "w": w,
    }


def xmss_sign(
    message: bytes,
    sk: dict,
    tree: list[list[bytes]],
) -> dict:
    """Sign with XMSS (simplified).

    :param message: Message bytes.
    :param sk: Secret key from xmss_keygen().
    :param tree: Merkle tree from xmss_keygen().
    :return: dict with signature, index, auth_path.
    """
    idx = sk["index"]
    wots_sk = sk["wots_keys"][idx]
    wots_sig = wots_sign(message, wots_sk)
    auth_path = merkle_auth_path(tree, idx)
    sk["index"] = idx + 1
    return {
        "wots_signature": wots_sig,
        "index": idx,
        "auth_path": auth_path,
    }
