"""Test ldpc_encode."""
import numpy as np

from moirais.fn._containers import CryptoResult
from moirais.fn.ldpce import ldpc_encode, ldpce


class TestLdpcEncode:
    def test_basic(self):
        G = np.eye(4, 7, dtype=np.uint8)
        message = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = ldpc_encode(G, message)
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "LDPC"
        assert result.operation == "encode"

    def test_extra_has_codeword(self):
        G = np.eye(3, 6, dtype=np.uint8)
        message = np.array([1, 1, 0], dtype=np.uint8)
        result = ldpc_encode(G, message)
        assert "codeword" in result.extra

    def test_alias(self):
        assert ldpce is ldpc_encode
