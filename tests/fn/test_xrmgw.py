"""Tests for morie.fn.xrmgw -- MGWR estimation"""

import numpy as np
import pytest

from morie.fn.xrmgw import mgwr_estimate


class TestMgwrEstimate:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mgwr_estimate(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = mgwr_estimate(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
