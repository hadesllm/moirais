"""Tests for morie.fn.srvfn -- Survival function from life table."""

import pytest
from morie.fn.srvfn import survival_function


class TestSurvivalFunction:
    def test_no_death(self):
        qx = [0.0, 0.0, 0.0]
        res = survival_function(qx)
        assert res.extra["S"][-1] == pytest.approx(1.0)

    def test_all_die(self):
        qx = [0.5, 0.5, 1.0]
        res = survival_function(qx)
        assert res.extra["lx"][-1] == pytest.approx(0.0)

    def test_invalid_qx(self):
        with pytest.raises(ValueError):
            survival_function([0.5, 1.5])
