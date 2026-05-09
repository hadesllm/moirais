"""Test fir_design (firds)."""
import numpy as np
from moirais.fn.firds import fir_design, firds
from moirais.fn._containers import DescriptiveResult


class TestFirDesign:
    def test_basic(self):
        result = fir_design(numtaps=31, cutoff=50.0, fs=500.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "fir_design"

    def test_coefficients(self):
        result = fir_design(numtaps=31, cutoff=50.0, fs=500.0)
        coeffs = result.extra["coefficients"]
        assert len(coeffs) == 31

    def test_alias(self):
        assert firds is fir_design
