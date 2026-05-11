"""Tests for morie.fn.svtut."""

import numpy as np
import pytest
from morie.fn.svtut import svtut


class TestSvtut:
    def test_basic(self):
        result = svtut(np.array([0.0, 0.0]), np.array([0.5, 0.5]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = svtut(np.array([0.0, 0.0]), np.array([0.5, 0.5]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svtut(np.array([0.0, 0.0]), np.array([0.5, 0.5]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = svtut(np.array([0.0, 0.0]), np.array([0.5, 0.5]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
