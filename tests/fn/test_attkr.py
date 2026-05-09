"""Tests for moirais.fn.attkr — Attack rate."""

import pytest

from moirais.fn.attkr import attack_rate


class TestAttackRate:
    def test_basic(self):
        res = attack_rate(50, 200)
        assert res.estimate == pytest.approx(0.25)
        assert res.ci_lower < 0.25 < res.ci_upper

    def test_zero_cases(self):
        res = attack_rate(0, 100)
        assert res.estimate == 0.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            attack_rate(10, 5)
