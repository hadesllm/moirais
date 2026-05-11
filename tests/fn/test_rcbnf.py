"""Tests for morie.fn.rcbnf."""

import numpy as np
import pytest
from morie.fn.rcbnf import rcbnf


class TestRcbnf:
    def test_basic(self):
        result = rcbnf(0.5, 0.0)
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = rcbnf(0.5, 0.0)
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = rcbnf(0.5, 0.0)
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = rcbnf(0.5, 0.0)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
