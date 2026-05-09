"""Tests for moirais.fn.frcst -- renewal equation forecast."""

import numpy as np
import pytest
from moirais.fn.frcst import renewal_forecast


class TestRenewalForecast:
    def test_stable_rt1(self):
        inc = np.ones(20) * 10
        si = np.array([0.3, 0.5, 0.2])
        res = renewal_forecast(inc, si, Rt=1.0, horizon=10)
        np.testing.assert_allclose(res["forecast"], 10.0, atol=1.0)

    def test_growing(self):
        inc = np.ones(20) * 10
        si = np.array([0.3, 0.5, 0.2])
        res = renewal_forecast(inc, si, Rt=2.0, horizon=10)
        assert res["forecast"][-1] > res["forecast"][0]

    def test_declining(self):
        inc = np.ones(20) * 100
        si = np.array([0.3, 0.5, 0.2])
        res = renewal_forecast(inc, si, Rt=0.5, horizon=10)
        assert res["forecast"][-1] < res["forecast"][0]

    def test_horizon_length(self):
        inc = np.ones(10) * 5
        si = np.array([0.5, 0.5])
        res = renewal_forecast(inc, si, Rt=1.0, horizon=7)
        assert len(res["forecast"]) == 7
        assert len(res["t_forecast"]) == 7

    def test_stochastic_pi(self):
        inc = np.ones(20) * 50
        si = np.array([0.2, 0.5, 0.3])
        res = renewal_forecast(inc, si, Rt=1.0, horizon=7, n_sim=200, seed=42)
        assert res["ci_lower"][0] <= res["forecast"][0]
        assert res["ci_upper"][0] >= res["forecast"][0]

    def test_negative_rt_raises(self):
        with pytest.raises(ValueError):
            renewal_forecast(np.ones(5), np.array([1.0]), Rt=-0.5)
