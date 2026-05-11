"""Test enob_compute (enob)."""
from morie.fn.enob import enob_compute, enob
from morie.fn._containers import DescriptiveResult


class TestEnob:
    def test_basic(self):
        result = enob_compute(50.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "enob"
        assert abs(result.value - (50.0 - 1.76) / 6.02) < 1e-10

    def test_16bit(self):
        sinad_16 = 6.02 * 16 + 1.76
        result = enob_compute(sinad_16)
        assert abs(result.value - 16.0) < 1e-10

    def test_alias(self):
        assert enob is enob_compute
