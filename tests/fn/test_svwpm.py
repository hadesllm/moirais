"""Tests for morie.fn.svwpm."""

import numpy as np
import pytest
from morie.fn.svwpm import svwpm


class TestSvwpm:
    def test_basic(self):
        result = svwpm(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = svwpm(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svwpm(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = svwpm(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
