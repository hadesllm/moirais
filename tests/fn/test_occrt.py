"""Tests for moirais.fn.occrt -- occupational injury rate."""

import pytest
from moirais.fn.occrt import occupational_injury_rate


class TestOccupationalInjuryRate:
    def test_basic(self):
        res = occupational_injury_rate(n_injuries=5, n_fte=100)
        assert res.estimate == pytest.approx(5.0)

    def test_ci(self):
        res = occupational_injury_rate(5, 100)
        assert res.ci_lower < res.estimate < res.ci_upper
