"""Tests for morie.fn.wvpub."""

import numpy as np
import pytest
from morie.fn.wvpub import wvpub


class TestWvpub:
    def test_basic(self):
        result = wvpub(np.array([3.0, 2.0, 2.0, 1.0]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = wvpub(np.array([3.0, 2.0, 2.0, 1.0]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = wvpub(np.array([3.0, 2.0, 2.0, 1.0]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = wvpub(np.array([3.0, 2.0, 2.0, 1.0]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
