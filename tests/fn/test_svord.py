"""Tests for moirais.fn.svord."""

import numpy as np
import pytest
from moirais.fn.svord import svord


class TestSvord:
    def test_basic(self):
        result = svord(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = svord(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svord(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = svord(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
