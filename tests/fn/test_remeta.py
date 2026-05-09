"""Tests for moirais.fn.remeta -- random-effects meta-analysis."""

import pytest
from moirais.fn.remeta import random_effects_meta


class TestRandomEffectsMeta:
    def test_three_studies(self):
        """Pool 3 studies with DerSimonian-Laird."""
        result = random_effects_meta(
            estimates=[0.5, 0.6, 0.4],
            standard_errors=[0.1, 0.15, 0.12],
        )
        assert 0.3 < result.estimate < 0.7
        assert result.ci_lower < result.estimate
        assert result.ci_upper > result.estimate

    def test_has_tau2_and_i2(self):
        result = random_effects_meta([0.5, 1.0, 0.2], [0.1, 0.2, 0.15])
        assert "tau_squared" in result.extra
        assert "I_squared" in result.extra
        assert result.extra["tau_squared"] >= 0
        assert 0 <= result.extra["I_squared"] <= 100

    def test_prediction_interval(self):
        result = random_effects_meta([0.5, 0.8, 0.3], [0.1, 0.2, 0.15])
        assert "prediction_interval_lower" in result.extra
        assert "prediction_interval_upper" in result.extra
