"""Tests for moirais.fn.pabrd."""

import numpy as np
import pytest
from moirais.fn.pabrd import pabrd


class TestPabrd:
    def test_basic(self):
        result = pabrd(np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = pabrd(np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = pabrd(np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = pabrd(np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
