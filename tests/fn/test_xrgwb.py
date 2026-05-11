"""Tests for morie.fn.xrgwb -- GWR bandwidth selection"""

import numpy as np
import pytest

from morie.fn.xrgwb import gwr_bandwidth


class TestGwrBandwidth:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gwr_bandwidth(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gwr_bandwidth(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
