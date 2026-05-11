"""Tests for morie.fn.nnh -- number needed to harm."""

import pytest
from morie.fn.nnh import number_needed_to_harm


class TestNNH:
    def test_returns_float(self):
        """NNH should return ESRes with numeric estimate."""
        result = number_needed_to_harm(a=20, b=80, c=10, d=90)
        assert result.measure == "NNH"
        assert isinstance(result.estimate, float)
        assert result.estimate > 0

    def test_same_as_nnt_magnitude(self):
        """NNH magnitude should match NNT for same inputs."""
        from morie.fn.nnt import number_needed_to_treat
        nnt = number_needed_to_treat(a=20, b=80, c=10, d=90)
        nnh = number_needed_to_harm(a=20, b=80, c=10, d=90)
        assert nnh.estimate == pytest.approx(nnt.estimate, rel=1e-10)

    def test_has_ci(self):
        result = number_needed_to_harm(a=30, b=70, c=10, d=90)
        assert result.ci_lower is not None
        assert result.ci_upper is not None
