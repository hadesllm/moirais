"""Test gen_parity_check."""
from moirais.fn._containers import DescriptiveResult
from moirais.fn.genpk import gen_parity_check, genpk


class TestGenParityCheck:
    def test_basic_hamming(self):
        result = gen_parity_check(n=7, k=4, code_type="hamming")
        assert isinstance(result, DescriptiveResult)
        assert result.value == 7.0

    def test_shapes(self):
        result = gen_parity_check(n=7, k=4, code_type="hamming")
        G = result.extra["G"]
        H = result.extra["H"]
        assert G.shape[0] == 4
        assert G.shape[1] == 7
        assert H.shape[1] == 7

    def test_alias(self):
        assert genpk is gen_parity_check
