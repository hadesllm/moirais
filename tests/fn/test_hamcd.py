"""Test hamming_code."""
import numpy as np

from moirais.fn._containers import CryptoResult
from moirais.fn.hamcd import hamcd, hamming_code


class TestHammingCode:
    def test_basic_encode(self):
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = hamming_code(data, r=3, mode="encode")
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "Hamming"
        assert result.operation == "encode"

    def test_roundtrip(self):
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        encoded = hamming_code(data, r=3, mode="encode")
        codeword = encoded.extra["codeword"]
        decoded = hamming_code(np.array(codeword, dtype=np.uint8), r=3, mode="decode")
        assert decoded.success is True
        assert decoded.operation == "decode"

    def test_alias(self):
        assert hamcd is hamming_code
