"""Tests for morie.fn.elcnv."""

import numpy as np
import pytest
from morie.fn.elcnv import elcnv


class TestElcnv:
    def test_basic(self):
        result = elcnv(data=np.random.default_rng(42).normal(0, 1, 100))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = elcnv(data=np.random.default_rng(42).normal(0, 1, 100))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = elcnv(data=np.random.default_rng(42).normal(0, 1, 100))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = elcnv(data=np.random.default_rng(42).normal(0, 1, 100))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
