"""Tests for morie.fn.md3dp."""

import numpy as np
import pytest
from morie.fn.md3dp import md3dp


class TestMd3dp:
    def test_basic(self):
        result = md3dp(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]]), np.array([0.5, 0.5]), np.array([0.0, 0.0]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = md3dp(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]]), np.array([0.5, 0.5]), np.array([0.0, 0.0]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = md3dp(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]]), np.array([0.5, 0.5]), np.array([0.0, 0.0]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = md3dp(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]]), np.array([0.5, 0.5]), np.array([0.0, 0.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
