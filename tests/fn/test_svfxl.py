"""Tests for moirais.fn.svfxl."""

import numpy as np
import pytest
from moirais.fn.svfxl import svfxl


class TestSvfxl:
    def test_basic(self):
        result = svfxl(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = svfxl(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svfxl(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = svfxl(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
