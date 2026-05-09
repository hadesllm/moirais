"""Tests for moirais.fn.pleri."""

import numpy as np
import pytest
from moirais.fn.pleri import pleri


class TestPleri:
    def test_basic(self):
        result = pleri(np.array([-1.0, -0.8, -0.5, 0.5, 0.8, 1.0]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = pleri(np.array([-1.0, -0.8, -0.5, 0.5, 0.8, 1.0]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = pleri(np.array([-1.0, -0.8, -0.5, 0.5, 0.8, 1.0]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = pleri(np.array([-1.0, -0.8, -0.5, 0.5, 0.8, 1.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
