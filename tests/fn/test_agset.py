"""Tests for morie.fn.agset."""

import numpy as np
import pytest
from morie.fn.agset import agset


class TestAgset:
    def test_basic(self):
        result = agset(np.array([-0.5, 0.0, 0.5]), 0.3, -1.0)
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = agset(np.array([-0.5, 0.0, 0.5]), 0.3, -1.0)
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = agset(np.array([-0.5, 0.0, 0.5]), 0.3, -1.0)
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = agset(np.array([-0.5, 0.0, 0.5]), 0.3, -1.0)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
