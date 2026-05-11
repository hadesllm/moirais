"""Tests for morie.fn.svout."""

import numpy as np
import pytest
from morie.fn.svout import svout


class TestSvout:
    def test_basic(self):
        result = svout(np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 1.5, 3.0]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = svout(np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 1.5, 3.0]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svout(np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 1.5, 3.0]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = svout(np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 1.5, 3.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
