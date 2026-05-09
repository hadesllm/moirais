"""Tests for moirais.fn.wvber."""

import numpy as np
import pytest
from moirais.fn.wvber import wvber


class TestWvber:
    def test_basic(self):
        result = wvber(np.array([3.0, 2.0, 2.0, 1.0]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = wvber(np.array([3.0, 2.0, 2.0, 1.0]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = wvber(np.array([3.0, 2.0, 2.0, 1.0]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = wvber(np.array([3.0, 2.0, 2.0, 1.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
