"""Test ntru_encrypt."""
from moirais.fn._containers import CryptoResult
from moirais.fn.ntru import ntru_keygen
from moirais.fn.ntruc import ntru_encrypt, ntruc


class TestNtruEncrypt:
    def test_basic(self):
        keys = ntru_keygen(n=11, q=32)
        msg = [0] * 11
        msg[0] = 1
        result = ntru_encrypt(msg, keys.extra["pk"], n=11, q=32)
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "NTRU"
        assert result.operation == "encrypt"

    def test_extra_has_ciphertext(self):
        keys = ntru_keygen(n=11, q=32)
        msg = [0] * 11
        msg[0] = 1
        result = ntru_encrypt(msg, keys.extra["pk"], n=11, q=32)
        assert "ciphertext" in result.extra

    def test_alias(self):
        assert ntruc is ntru_encrypt
