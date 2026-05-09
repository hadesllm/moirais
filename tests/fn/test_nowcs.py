"""Tests for moirais.fn.nowcs -- nowcasting."""

import numpy as np
import pytest
from moirais.fn.nowcs import nowcast


class TestNowcast:
    def test_full_reporting(self):
        rep = np.array([10.0, 20.0, 30.0, 25.0, 15.0])
        delay = np.array([0.0, 0.2, 0.5, 0.3])
        res = nowcast(rep, delay)
        np.testing.assert_allclose(res["nowcast"][:2], rep[:2], atol=0.1)

    def test_recent_inflated(self):
        rep = np.array([100.0, 100.0, 100.0, 50.0, 20.0])
        delay = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
        res = nowcast(rep, delay)
        assert res["nowcast"][-1] > rep[-1]

    def test_completeness_decreasing(self):
        rep = np.ones(10) * 50
        delay = np.array([0.0, 0.2, 0.3, 0.3, 0.2])
        res = nowcast(rep, delay)
        comp = res["reporting_completeness"]
        assert comp[-1] <= comp[0]

    def test_inflation_factor(self):
        rep = np.array([100.0, 100.0, 100.0])
        delay = np.array([0.5, 0.5])
        res = nowcast(rep, delay)
        assert res["inflation_factor"][-1] >= 1.0

    def test_negative_pmf_raises(self):
        with pytest.raises(ValueError):
            nowcast(np.array([10.0]), np.array([-0.5, 0.5]))
