"""Tests for moirais.fn.odose — OTIS dose-response."""

import pytest
import numpy as np
from moirais.fn.odose import otis_dose_response
from moirais.fn._containers import DescriptiveResult


class TestOtisDoseResponse:

    def test_returns_descriptive(self):
        doses = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        outcomes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        result = otis_dose_response(doses, outcomes)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["slope"] > 0

    def test_no_trend(self):
        doses = np.array([1, 2, 3, 4])
        outcomes = np.array([5.0, 5.0, 5.0, 5.0])
        result = otis_dose_response(doses, outcomes)
        assert result.extra["slope"] == pytest.approx(0.0, abs=1e-10)
