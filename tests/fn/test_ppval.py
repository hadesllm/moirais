"""Tests for morie.fn.ppval."""

import numpy as np
import pytest
from morie.fn.ppval import ppval


class TestPpval:
    def test_basic(self):
        result = ppval(np.array([-0.5, 0.5]), np.random.default_rng(42).normal(0, 1, 50))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = ppval(np.array([-0.5, 0.5]), np.random.default_rng(42).normal(0, 1, 50))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = ppval(np.array([-0.5, 0.5]), np.random.default_rng(42).normal(0, 1, 50))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = ppval(np.array([-0.5, 0.5]), np.random.default_rng(42).normal(0, 1, 50))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
