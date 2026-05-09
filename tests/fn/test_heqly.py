"""Tests for moirais.fn.heqly -- QALY."""

import pytest
from moirais.fn.heqly import quality_adjusted_ly


class TestQALY:
    def test_basic(self):
        res = quality_adjusted_ly(utilities=[0.8, 0.6], durations=[5, 3])
        assert res.estimate == pytest.approx(0.8 * 5 + 0.6 * 3)

    def test_perfect_health(self):
        res = quality_adjusted_ly([1.0], [10])
        assert res.estimate == pytest.approx(10.0)

    def test_invalid_utility(self):
        with pytest.raises(ValueError):
            quality_adjusted_ly([1.5], [1])
