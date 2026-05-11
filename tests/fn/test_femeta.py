"""Tests for morie.fn.femeta -- fixed-effects meta-analysis."""

import pytest
from morie.fn.femeta import fixed_effects_meta


class TestFixedEffectsMeta:
    def test_three_studies(self):
        """Pool 3 studies with known effects."""
        result = fixed_effects_meta(
            estimates=[0.5, 0.6, 0.4],
            standard_errors=[0.1, 0.15, 0.12],
        )
        assert result.measure == "Fixed-effects meta-analysis"
        assert 0.3 < result.estimate < 0.7
        assert result.ci_lower < result.estimate
        assert result.ci_upper > result.estimate

    def test_identical_studies(self):
        """Identical studies should pool to same value."""
        result = fixed_effects_meta(
            estimates=[1.0, 1.0, 1.0],
            standard_errors=[0.1, 0.1, 0.1],
        )
        assert result.estimate == pytest.approx(1.0, abs=1e-10)

    def test_has_q_statistic(self):
        result = fixed_effects_meta([0.5, 0.8, 0.3], [0.1, 0.2, 0.15])
        assert "Q" in result.extra
        assert "Q_p_value" in result.extra
        assert result.extra["Q"] >= 0
