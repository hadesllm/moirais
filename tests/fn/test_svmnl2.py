"""Tests for morie.fn.svmnl2."""

import numpy as np
import pytest
from morie.fn.svmnl2 import svmnl2


class TestSvmnl2:
    def test_basic(self):
        result = svmnl2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0]]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = svmnl2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0]]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svmnl2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0]]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = svmnl2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0]]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
        assert result.name == "Multinomial Logit Vote Choice"
