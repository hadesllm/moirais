"""Tests for morie.fn.isvlt."""

import numpy as np
import pytest
from morie.fn.isvlt import isvlt


class TestIsvlt:
    def test_basic(self):
        result = isvlt(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = isvlt(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = isvlt(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = isvlt(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
