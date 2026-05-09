"""Tests for moirais.fn.mpstr."""

import numpy as np
import pytest
from moirais.fn.mpstr import mpstr


class TestMpstr:
    def test_basic(self):
        result = mpstr(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = mpstr(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = mpstr(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = mpstr(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
