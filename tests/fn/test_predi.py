"""Tests for moirais.fn.predi -- prediction interval from random-effects meta."""

import pytest
from moirais.fn.predi import prediction_interval


class TestPredictionInterval:
    def test_returns_tuple(self):
        """Should return (lower, upper) tuple."""
        lo, hi = prediction_interval(
            estimates=[0.5, 0.6, 0.4],
            standard_errors=[0.1, 0.15, 0.12],
        )
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi

    def test_wider_than_ci(self):
        """Prediction interval should be wider than the pooled CI."""
        from moirais.fn.remeta import random_effects_meta
        ests = [0.5, 0.8, 0.3]
        ses = [0.1, 0.2, 0.15]
        result = random_effects_meta(ests, ses)
        lo, hi = prediction_interval(ests, ses)
        assert lo <= result.ci_lower
        assert hi >= result.ci_upper

    def test_contains_pooled(self):
        """Prediction interval should contain the pooled estimate."""
        from moirais.fn.remeta import random_effects_meta
        ests = [0.5, 0.6, 0.7]
        ses = [0.1, 0.1, 0.1]
        pooled = random_effects_meta(ests, ses).estimate
        lo, hi = prediction_interval(ests, ses)
        assert lo < pooled < hi
