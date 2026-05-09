"""Tests for moirais.fn.excmt -- excess mortality."""

import numpy as np
import pytest
from moirais.fn.excmt import excess_mortality


class TestExcessMortality:
    def test_no_excess(self):
        obs = np.array([100.0, 100.0, 100.0])
        base = np.array([100.0, 100.0, 100.0])
        res = excess_mortality(obs, base)
        assert res["total_excess"] == pytest.approx(0.0)
        assert res["p_score_pct"] == pytest.approx(0.0)

    def test_positive_excess(self):
        obs = np.array([120.0, 130.0, 110.0])
        base = np.array([100.0, 100.0, 100.0])
        res = excess_mortality(obs, base)
        assert res["total_excess"] == pytest.approx(60.0)
        assert res["p_score_pct"] == pytest.approx(20.0)

    def test_excess_per_period(self):
        obs = np.array([150.0, 80.0])
        base = np.array([100.0, 100.0])
        res = excess_mortality(obs, base)
        np.testing.assert_allclose(res["excess_per_period"], [50.0, -20.0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            excess_mortality(np.array([1, 2]), np.array([1, 2, 3]))

    def test_significant_periods(self):
        obs = np.array([100.0, 200.0, 100.0])
        base = np.array([100.0, 100.0, 100.0])
        res = excess_mortality(obs, base)
        assert 1 in res["significant_periods"]
