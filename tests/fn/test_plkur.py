"""Tests for moirais.fn.plkur."""

import numpy as np
import pytest
from moirais.fn.plkur import plkur


class TestPlkur:
    def test_basic(self):
        result = plkur(np.array([-1.0, -0.8, -0.5, 0.5, 0.8, 1.0]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = plkur(np.array([-1.0, -0.8, -0.5, 0.5, 0.8, 1.0]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = plkur(np.array([-1.0, -0.8, -0.5, 0.5, 0.8, 1.0]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = plkur(np.array([-1.0, -0.8, -0.5, 0.5, 0.8, 1.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
