"""Tests for moirais.fn.dtrot."""

import numpy as np
import pytest
from moirais.fn.dtrot import dtrot


class TestDtrot:
    def test_basic(self):
        result = dtrot(np.random.default_rng(42).normal(0, 1, (50, 5)))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = dtrot(np.random.default_rng(42).normal(0, 1, (50, 5)))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = dtrot(np.random.default_rng(42).normal(0, 1, (50, 5)))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = dtrot(np.random.default_rng(42).normal(0, 1, (50, 5)))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
