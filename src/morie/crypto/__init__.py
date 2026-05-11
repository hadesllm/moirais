"""MORIE post-quantum cryptography (research/educational).

WARNING: This is a pure-Python reference implementation of NIST-standardized
post-quantum cryptographic primitives (ML-KEM-768, ChaCha20-Poly1305).
It is NOT constant-time and NOT suitable for production secrets. For
production-grade encryption, use hardware KMS or audited C implementations.

Pure-Python ML-KEM is ~100x slower than the reference C implementation.
"""

from morie.crypto._chacha import chacha20_poly1305_decrypt, chacha20_poly1305_encrypt
from morie.crypto._gf2m import (
    find_irreducible,
    gf2_matrix_add,
    gf2_matrix_inv,
    gf2_matrix_mul,
    gf2m_add,
    gf2m_inv,
    gf2m_mul,
    gf2m_pow,
)
from morie.crypto._kdf import hkdf_sha256
from morie.crypto._lattice_core import (
    babai_nearest_plane,
    bkz_reduce,
    gram_schmidt,
    lll_reduce,
    lwe_key_exchange,
    lwe_sample,
    rlwe_sample,
    svp_approx,
)
from morie.crypto._mlkem import mlkem768_decaps, mlkem768_encaps, mlkem768_keygen
from morie.crypto._poly_ring import (
    build_zetas,
    inv_ntt,
    ntt,
    poly_add,
    poly_mul_ntt,
    poly_ring_mul,
    poly_ring_mul_xn_minus_1,
    poly_sub,
)
from morie.crypto.hybrid import hybrid_decrypt, hybrid_encrypt
from morie.crypto.hybrid import keygen as hybrid_keygen
from morie.crypto.keystore import create_keystore, list_keys, load_keypair, store_keypair

try:
    from morie.crypto._dilithium import mldsa_keygen, mldsa_sign, mldsa_verify
except Exception:
    pass
try:
    from morie.crypto._ntru import ntru_decrypt, ntru_encrypt, ntru_keygen
except Exception:
    pass
try:
    from morie.crypto._mceliece import mceliece_decaps, mceliece_encaps, mceliece_keygen
except Exception:
    pass

__all__ = [
    "mlkem768_keygen",
    "mlkem768_encaps",
    "mlkem768_decaps",
    "chacha20_poly1305_encrypt",
    "chacha20_poly1305_decrypt",
    "hkdf_sha256",
    "hybrid_encrypt",
    "hybrid_decrypt",
    "hybrid_keygen",
    "create_keystore",
    "store_keypair",
    "load_keypair",
    "list_keys",
    "mldsa_keygen",
    "mldsa_sign",
    "mldsa_verify",
    "ntru_keygen",
    "ntru_encrypt",
    "ntru_decrypt",
    "mceliece_keygen",
    "mceliece_encaps",
    "mceliece_decaps",
    "ntt",
    "inv_ntt",
    "build_zetas",
    "poly_add",
    "poly_sub",
    "poly_mul_ntt",
    "poly_ring_mul",
    "poly_ring_mul_xn_minus_1",
    "find_irreducible",
    "gf2m_add",
    "gf2m_mul",
    "gf2m_inv",
    "gf2m_pow",
    "gf2_matrix_add",
    "gf2_matrix_mul",
    "gf2_matrix_inv",
    "gram_schmidt",
    "lll_reduce",
    "bkz_reduce",
    "babai_nearest_plane",
    "svp_approx",
    "lwe_sample",
    "rlwe_sample",
    "lwe_key_exchange",
]
