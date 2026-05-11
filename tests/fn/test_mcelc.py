"""Test mceliece_encrypt."""
from morie.fn._containers import CryptoResult
from morie.fn.mcelc import mcelc, mceliece_encrypt


class TestMcelieceEncrypt:
    def test_basic(self):
        result = mceliece_encrypt(message=b"test")
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "McEliece"
        assert result.operation == "encapsulate"

    def test_extra_has_shared_secret(self):
        result = mceliece_encrypt(message=b"test")
        assert "shared_secret" in result.extra
        assert "syndrome" in result.extra
        assert "pk" in result.extra

    def test_alias(self):
        assert mcelc is mceliece_encrypt
