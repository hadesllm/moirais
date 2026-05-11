"""Tests for morie.fn.xrwis -- Weights islands detection"""

import numpy as np
import pytest

from morie.fn.xrwis import w_islands


class TestWIslands:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_islands(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_islands(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
