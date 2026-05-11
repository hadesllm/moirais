"""Tests for morie.fn.svexp2."""

import numpy as np
import pytest
from morie.fn.svexp2 import svexp2


class TestSvexp2:
    def test_basic(self):
        result = svexp2(np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = svexp2(np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svexp2(np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = svexp2(np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
        assert result.name == "Exponential Vote Probability Decay"
