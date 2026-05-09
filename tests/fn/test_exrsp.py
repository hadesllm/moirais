"""Tests for moirais.fn.exrsp — exposure-response modeling."""
import numpy as np
import pytest
from moirais.fn.exrsp import exposure_response


class TestExposureResponse:
    def test_linear_exposure(self):
        rng = np.random.default_rng(42)
        exposure = rng.uniform(0, 10, 200)
        outcome = 2.0 * exposure + rng.normal(0, 1, 200)
        res = exposure_response(exposure, outcome)
        assert res.extra["r_squared"] > 0.5

    def test_knots_count(self):
        rng = np.random.default_rng(42)
        exposure = rng.uniform(0, 10, 150)
        outcome = exposure ** 0.5 + rng.normal(0, 0.5, 150)
        res = exposure_response(exposure, outcome, n_spline_knots=5)
        assert len(res.extra["knots"]) == 5

    def test_predicted_shape(self):
        rng = np.random.default_rng(42)
        exposure = rng.uniform(1, 20, 100)
        outcome = np.log(exposure) + rng.normal(0, 0.3, 100)
        res = exposure_response(exposure, outcome)
        assert "predicted_response" in res.extra
