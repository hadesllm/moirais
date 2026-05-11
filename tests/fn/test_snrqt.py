"""Test snr_quantization (snrqt)."""
from morie.fn.snrqt import snr_quantization, snrqt
from morie.fn._containers import DescriptiveResult


class TestSnrqt:
    def test_basic(self):
        result = snr_quantization(8)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "snr_quantization"
        assert abs(result.value - (6.02 * 8 + 1.76)) < 1e-10

    def test_16bit(self):
        result = snr_quantization(16)
        assert abs(result.value - 98.08) < 1e-10

    def test_alias(self):
        assert snrqt is snr_quantization
