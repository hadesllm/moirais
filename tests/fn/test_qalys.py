"""Tests for moirais.fn.qalys -- QALY computation."""

import pytest
from moirais.fn.qalys import qaly_computation


class TestQALY:
    def test_perfect_health(self):
        res = qaly_computation(durations=[10.0], utilities=[1.0], discount_rate=0.0)
        assert res.estimate == pytest.approx(10.0)

    def test_half_utility(self):
        res = qaly_computation(durations=[10.0], utilities=[0.5], discount_rate=0.0)
        assert res.estimate == pytest.approx(5.0)

    def test_invalid_utility(self):
        with pytest.raises(ValueError):
            qaly_computation(durations=[10], utilities=[1.5])
