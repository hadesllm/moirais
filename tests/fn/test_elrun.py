"""Tests for morie.fn.elrun."""

import numpy as np
import pytest
from morie.fn.elrun import elrun


class TestElrun:
    def test_basic(self):
        result = elrun(data=np.random.default_rng(42).normal(0, 1, 100))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = elrun(data=np.random.default_rng(42).normal(0, 1, 100))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = elrun(data=np.random.default_rng(42).normal(0, 1, 100))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = elrun(data=np.random.default_rng(42).normal(0, 1, 100))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
