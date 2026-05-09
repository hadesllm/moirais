"""Tests for moirais.fn.svmxl2."""

import numpy as np
import pytest
from moirais.fn.svmxl2 import svmxl2


class TestSvmxl2:
    def test_basic(self):
        result = svmxl2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), n_draws=10)
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = svmxl2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), n_draws=10)
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svmxl2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), n_draws=10)
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = svmxl2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), n_draws=10)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
        assert result.name == "Mixed Logit Spatial Model"
