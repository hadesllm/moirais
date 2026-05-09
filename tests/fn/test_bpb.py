"""Test bits_per_byte."""
import numpy as np
from moirais.fn.bpb import bits_per_byte, bpb
from moirais.fn._containers import DescriptiveResult


class TestBitsPerByte:
    def test_basic(self):
        result = bits_per_byte(1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bits_per_byte"

    def test_value(self):
        result = bits_per_byte(np.log(2))
        assert abs(result.value - 1.0) < 1e-10

    def test_zero_loss(self):
        result = bits_per_byte(0.0)
        assert abs(result.value) < 1e-10

    def test_base_2(self):
        result = bits_per_byte(1.5, base="2")
        assert abs(result.value - 1.5) < 1e-10

    def test_alias(self):
        assert bpb is bits_per_byte
