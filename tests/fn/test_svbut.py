"""Tests for morie.fn.svbut."""

import numpy as np
import pytest
from morie.fn.svbut import svbut


class TestSvbut:
    def test_basic(self):
        result = svbut(np.array([0.0]), np.array([0.5]), np.array([2.0]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = svbut(np.array([0.0]), np.array([0.5]), np.array([2.0]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svbut(np.array([0.0]), np.array([0.5]), np.array([2.0]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = svbut(np.array([0.0]), np.array([0.5]), np.array([2.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
