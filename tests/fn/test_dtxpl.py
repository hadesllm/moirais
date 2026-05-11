"""Tests for morie.fn.dtxpl."""

import numpy as np
import pytest
from morie.fn.dtxpl import dtxpl


class TestDtxpl:
    def test_basic(self):
        result = dtxpl(np.random.default_rng(42).normal(0, 1, (50, 5)))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = dtxpl(np.random.default_rng(42).normal(0, 1, (50, 5)))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = dtxpl(np.random.default_rng(42).normal(0, 1, (50, 5)))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = dtxpl(np.random.default_rng(42).normal(0, 1, (50, 5)))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
