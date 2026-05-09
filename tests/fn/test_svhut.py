"""Tests for moirais.fn.svhut."""

import numpy as np
import pytest
from moirais.fn.svhut import svhut


class TestSvhut:
    def test_basic(self):
        result = svhut(np.array([0.0, 1.0]), np.array([0.5, 0.5]), np.array([0.7, 0.3]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = svhut(np.array([0.0, 1.0]), np.array([0.5, 0.5]), np.array([0.7, 0.3]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svhut(np.array([0.0, 1.0]), np.array([0.5, 0.5]), np.array([0.7, 0.3]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = svhut(np.array([0.0, 1.0]), np.array([0.5, 0.5]), np.array([0.7, 0.3]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
