"""Tests for morie.fn.paf -- Population Attributable Fraction."""

import pytest
from morie.fn.paf import population_attributable_fraction


class TestPAF:
    def test_known_paf(self):
        """pe=0.3, RR=2 -> PAF = 0.3*1 / (1 + 0.3*1) = 0.3/1.3."""
        result = population_attributable_fraction(0.3, 2.0)
        assert result["paf"] == pytest.approx(0.3 / 1.3, rel=1e-6)

    def test_rr_one_paf_zero(self):
        """RR=1 means no effect, PAF should be 0."""
        result = population_attributable_fraction(0.5, 1.0)
        assert result["paf"] == pytest.approx(0.0)

    def test_with_ci(self):
        result = population_attributable_fraction(0.4, 2.5, ci_rr=(1.5, 4.0))
        assert result["ci_lower"] is not None
        assert result["ci_upper"] is not None
        assert result["ci_lower"] < result["paf"] < result["ci_upper"]

    def test_invalid_prevalence_raises(self):
        with pytest.raises(ValueError, match="prevalence"):
            population_attributable_fraction(1.5, 2.0)
