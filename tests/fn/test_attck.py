"""Tests for morie.fn.attck -- attack rate and secondary attack rate."""

import numpy as np
import pytest
from morie.fn.attck import attack_rate, secondary_attack_rate


class TestAttackRate:
    def test_basic(self):
        res = attack_rate(50, 1000)
        assert res["attack_rate"] == pytest.approx(0.05)

    def test_ci_contains_rate(self):
        res = attack_rate(100, 1000)
        assert res["ci_lower"] < 0.10 < res["ci_upper"]

    def test_zero_cases(self):
        res = attack_rate(0, 1000)
        assert res["attack_rate"] == 0.0
        assert abs(res["ci_lower"]) < 1e-10

    def test_exceeds_pop_raises(self):
        with pytest.raises(ValueError):
            attack_rate(1001, 1000)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            attack_rate(-1, 100)


class TestSecondaryAttackRate:
    def test_basic(self):
        res = secondary_attack_rate(3, 10)
        assert res["sar"] == pytest.approx(0.3)

    def test_ci(self):
        res = secondary_attack_rate(5, 20)
        assert res["ci_lower"] < 0.25 < res["ci_upper"]

    def test_zero_secondary(self):
        res = secondary_attack_rate(0, 15)
        assert res["sar"] == 0.0
        assert abs(res["ci_lower"]) < 1e-10

    def test_all_infected(self):
        res = secondary_attack_rate(10, 10)
        assert res["sar"] == pytest.approx(1.0)
        assert res["ci_upper"] == 1.0
