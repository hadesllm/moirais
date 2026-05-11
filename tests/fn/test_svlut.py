"""Tests for morie.fn.svlut."""

import numpy as np
import pytest
from morie.fn.svlut import svlut


class TestSvlut:
    def test_basic(self):
        result = svlut(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = svlut(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svlut(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = svlut(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
