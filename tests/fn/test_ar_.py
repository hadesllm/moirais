"""Tests for moirais.fn.ar_ -- Attack rate (cumulative incidence)."""

import pytest
from moirais.fn.ar_ import attack_rate


class TestAttackRate:
    def test_known_rate(self):
        result = attack_rate(20, 100)
        assert result["rate"] == pytest.approx(0.20)

    def test_ci_in_zero_one(self):
        result = attack_rate(15, 200)
        assert 0 <= result["ci_lower"] <= result["ci_upper"] <= 1

    def test_zero_cases(self):
        result = attack_rate(0, 100)
        assert result["rate"] == 0.0

    def test_cases_exceed_pop_raises(self):
        with pytest.raises(ValueError, match="exceed"):
            attack_rate(101, 100)
