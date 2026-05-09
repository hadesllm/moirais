"""Tests for moirais.fn.xrmgb -- MGWR variable bandwidths"""

import numpy as np
import pytest

from moirais.fn.xrmgb import mgwr_bandwidths


class TestMgwrBandwidths:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mgwr_bandwidths(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = mgwr_bandwidths(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
