"""Tests for moirais.fn.mhtrn -- mental health trend."""

import pytest
from moirais.fn.mhtrn import mental_health_trend


class TestMentalHealthTrend:
    def test_increasing(self):
        res = mental_health_trend([0.10, 0.12, 0.14], [2020, 2021, 2022])
        assert res.value > 0

    def test_flat(self):
        res = mental_health_trend([0.10, 0.10], [2020, 2021])
        assert res.value == pytest.approx(0.0, abs=1e-10)
