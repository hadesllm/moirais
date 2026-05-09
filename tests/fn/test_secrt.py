"""Tests for moirais.fn.secrt -- Secondary attack rate."""

import pytest
from moirais.fn.secrt import secondary_attack_rate


class TestSecondaryAttackRate:
    def test_known(self):
        res = secondary_attack_rate(10, 50)
        assert res.measure == "SAR"
        assert res.estimate == pytest.approx(0.2)

    def test_ci(self):
        res = secondary_attack_rate(10, 50)
        assert res.ci_lower <= res.estimate
        assert res.ci_upper >= res.estimate

    def test_invalid(self):
        with pytest.raises(ValueError):
            secondary_attack_rate(10, 0)
