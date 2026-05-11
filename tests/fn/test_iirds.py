"""Test iir_design (iirds)."""
import numpy as np
from morie.fn.iirds import iir_design, iirds
from morie.fn._containers import DescriptiveResult


class TestIirDesign:
    def test_basic(self):
        result = iir_design(order=4, cutoff=50.0, fs=500.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "iir_design"

    def test_coefficients(self):
        result = iir_design(order=4, cutoff=50.0, fs=500.0)
        assert "b" in result.extra
        assert "a" in result.extra

    def test_alias(self):
        assert iirds is iir_design
