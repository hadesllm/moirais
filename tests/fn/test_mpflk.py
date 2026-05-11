"""Tests for morie.fn.mpflk."""

import numpy as np
import pytest
from morie.fn.mpflk import mpflk


class TestMpflk:
    def test_basic(self):
        result = mpflk(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = mpflk(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = mpflk(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = mpflk(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
