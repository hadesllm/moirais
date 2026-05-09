"""Tests for moirais.fn.zeflx -- Flexible spatial scan (Tango)"""

import numpy as np
import pytest

from moirais.fn.zeflx import flexible_scan


class TestFlexibleScan:
    def test_basic(self):
        observed = np.array([5, 3, 8, 2, 10])
        result = flexible_scan(observed)
        assert result.statistic is not None

    def test_output_type(self):
        result = flexible_scan(np.array([1,2,3,4,5]))
        assert hasattr(result, "statistic")
