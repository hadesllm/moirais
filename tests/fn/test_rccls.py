"""Tests for morie.fn.rccls."""

import numpy as np
import pytest
from morie.fn.rccls import rccls


class TestRccls:
    def test_basic(self):
        result = rccls(0.5, 0.0)
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = rccls(0.5, 0.0)
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = rccls(0.5, 0.0)
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = rccls(0.5, 0.0)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
