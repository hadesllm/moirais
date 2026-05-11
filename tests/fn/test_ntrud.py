"""Test ntru_decrypt."""
from morie.fn._containers import CryptoResult
from morie.fn.ntru import ntru_keygen
from morie.fn.ntruc import ntru_encrypt
from morie.fn.ntrud import ntru_decrypt, ntrud


class TestNtruDecrypt:
    def test_roundtrip(self):
        keys = ntru_keygen(n=11, q=32)
        msg = [0] * 11
        msg[0] = 1
        encrypted = ntru_encrypt(msg, keys.extra["pk"], n=11, q=32)
        result = ntru_decrypt(
            encrypted.extra["ciphertext"],
            keys.extra["sk"],
            n=11,
            q=32,
        )
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "NTRU"
        assert result.operation == "decrypt"

    def test_alias(self):
        assert ntrud is ntru_decrypt
