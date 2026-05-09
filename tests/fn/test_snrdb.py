"""Test snr_to_linear (snrdb)."""
from moirais.fn.snrdb import snr_to_linear, snrdb
from moirais.fn._containers import DescriptiveResult


class TestSNRToLinear:
    def test_zero_db(self):
        result = snr_to_linear(0.0)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 1.0) < 1e-10

    def test_ten_db(self):
        assert abs(snr_to_linear(10.0).value - 10.0) < 1e-10

    def test_twenty_db(self):
        assert abs(snr_to_linear(20.0).value - 100.0) < 1e-8

    def test_alias(self):
        assert snrdb is snr_to_linear
