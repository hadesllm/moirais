"""Test ldpc_generate."""
from morie.fn._containers import DescriptiveResult
from morie.fn.ldpcg import ldpc_generate, ldpcg


class TestLdpcGenerate:
    def test_basic(self):
        result = ldpc_generate(n=20)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 20.0

    def test_H_shape(self):
        result = ldpc_generate(n=20, rate=0.5)
        H = result.extra["H"]
        assert H.shape[1] == 20
        assert H.shape[0] == 10

    def test_alias(self):
        assert ldpcg is ldpc_generate
