"""Test irreducible_poly."""
import numpy as np
import pytest
from moirais.fn.irpol import irreducible_poly
from moirais.fn._containers import DescriptiveResult


class TestIrreduciblePoly:
    def test_basic(self):
        result = irreducible_poly(m=8)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        result = irreducible_poly(m=8)
        assert "polynomial" in result.extra

    def test_degree(self):
        result = irreducible_poly(m=8)
        assert "degree" in result.extra
        assert result.extra["degree"] == 8
