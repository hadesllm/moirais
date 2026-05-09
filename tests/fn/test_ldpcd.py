"""Test ldpc_decode."""
import numpy as np

from moirais.fn._containers import CryptoResult
from moirais.fn.ldpcd import ldpc_decode, ldpcd


class TestLdpcDecode:
    def test_basic(self):
        H = np.array([
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ], dtype=np.uint8)
        received = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8)
        result = ldpc_decode(H, received)
        assert isinstance(result, CryptoResult)
        assert result.algorithm == "LDPC"
        assert result.operation == "decode"

    def test_zero_syndrome_success(self):
        H = np.array([
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ], dtype=np.uint8)
        received = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8)
        result = ldpc_decode(H, received)
        assert result.success is True

    def test_alias(self):
        assert ldpcd is ldpc_decode
