"""Tests for moirais.fn.isfrm."""

import numpy as np
import pytest
from moirais.fn.isfrm import isfrm


class TestIsfrm:
    def test_basic(self):
        result = isfrm(np.array([0.5, -0.3, 0.8]), np.array([0.5, 0.3, 0.2]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = isfrm(np.array([0.5, -0.3, 0.8]), np.array([0.5, 0.3, 0.2]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = isfrm(np.array([0.5, -0.3, 0.8]), np.array([0.5, 0.3, 0.2]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = isfrm(np.array([0.5, -0.3, 0.8]), np.array([0.5, 0.3, 0.2]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
