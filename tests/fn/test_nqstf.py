"""Test nyquist_freq (nqstf)."""
from moirais.fn.nqstf import nyquist_freq, nqstf
from moirais.fn._containers import DescriptiveResult


class TestNqstf:
    def test_basic(self):
        result = nyquist_freq(1000.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "nyquist_freq"
        assert result.value == 500.0

    def test_known_value(self):
        result = nyquist_freq(44100.0)
        assert result.value == 22050.0

    def test_alias(self):
        assert nqstf is nyquist_freq
