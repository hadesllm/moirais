"""Tests for morie.fn.hsent."""

import numpy as np
import pytest
from morie.fn.hsent import hsent


class TestHsent:
    def test_basic(self):
        result = hsent(-0.3, 0.3)
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = hsent(-0.3, 0.3)
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = hsent(-0.3, 0.3)
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = hsent(-0.3, 0.3)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
